import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import List, Union, Any

from api.app_server import AppServer
from api.app_server import serve as grpc_serve
from api.common.typedef import InferCache
from app.detector import Detector
import app.proto.edge_pb2 as pb2
import app.proto.edge_pb2_grpc as pb2_grpc
import grpc

OUT_DIR = os.path.join(
    Path(Path(__file__).parent.resolve()).parents[1], "server-res")

# header len for the packet
HEADER_SZ = 28
HEADER_PAYLOAD_SZ = 16
HEADER_SEG_IDX_SZ = 12
PACKET_SZ = 4096

# if output the inference result as image
TO_IMG = False

# logger set up
logger = logging.getLogger("Pipeline")
handler = logging.StreamHandler()
format = logging.Formatter("[%(name)s][%(levelname)s] - %(message)s")
handler.setFormatter(format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_img(vpath: str, seg_idx: int, app_idx: int) -> bool:
    out_path = os.path.join(OUT_DIR, str(app_idx),
                            f"{seg_idx}-%d.png")

    compress_res = subprocess.run(["ffmpeg", "-y", "-i", vpath, out_path],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True
                                  )

    if compress_res.returncode != 0:
        logger.error("DECODING FAILED")
        logger.error(compress_res.stdout)
        logger.error(compress_res.stderr)
        exit(os.EX_DATAERR)
    return True


class Server(AppServer):

    def __init__(self, client_uri: str, need_concierge: bool, data_port: int,
                 detector: Detector, app_idx, control_port: int) -> None:
        super().__init__(client_uri, need_concierge, control_port)
        self.detector: Detector = detector
        self.port: int = data_port
        self.app_idx: int = app_idx
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(10)
        self.p_executor: ProcessPoolExecutor = ProcessPoolExecutor(2)
        self.client_init: bool = False
        self.current_bw: int = 0
        self.edge_stub: Union[None, pb2_grpc.EdgeStub] = None
        self.executor.submit(self.detector.run)

    def serve(self):
        # init socket and wait for requests
        skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        addr = ("0.0.0.0", self.port)
        skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        skt.bind(addr)
        skt.listen(5)
        logger.info(f"[App{self.app_idx}] Server is listening on port"
                    f" {self.port}")

        tid = 0
        while True:
            conn, addr = skt.accept()
            if not self.client_init:
                edge_grpc_uri = f"{self.client_uri}:{20000 + self.app_idx}"
                channel = grpc.insecure_channel(edge_grpc_uri)
                self.edge_stub = pb2_grpc.EdgeStub(channel)
                resp = self.edge_stub.UpdateBitrate(
                        pb2.Bitrate(bitrate=self.current_bw))
                if resp.updated:
                    logger.info(f"[App{self.app_idx}] Connected Edge GRPC"
                                f" {edge_grpc_uri}")
                    logger.info(f"[App{self.app_idx}] Edge updated its"
                                " bitrate")
                self.client_init = True
            serving_t = ServingThread(tid, self.detector, conn, addr, self)
            self.executor.submit(serving_t.run)
            tid += 1

    def default_callback(self, is_succed: bool, actual_amount: float):
        if is_succed and actual_amount != 0:
            self.current_bw = int(actual_amount)
            if self.edge_stub is not None:
                resp = self.edge_stub.UpdateBitrate(
                        pb2.Bitrate(bitrate=int(self.current_bw)))
                if resp.updated:
                    logger.info(f"[App{self.app_idx}] Edge updated its"
                                " bitrate")

    def run_inference(self, input) -> Any:
        return super().run_inference(input)

    def check_accuracy(self, inference, gt) -> float:
        return super().check_accuracy(inference, gt)

    def job_done(self, seg_idx):
        if self.edge_stub is not None:
            self.edge_stub.NotifyJobDone(
                pb2.IsJobDone(done=True, seg_idx=seg_idx))


class ServingThread():

    def __init__(self, tid: int, detector: Detector, conn: socket.socket,
                 addr: tuple, server: Server):
        self.conn: socket.socket = conn
        self.detector: Detector = detector
        self.tid: int = tid
        self.addr: tuple = addr
        self.server = server
        self.seg_idx: int
        self.remaining_sz: int

    def _recv_header(self):
        # according to the protocol, the header indicates the total video size
        header = bytearray(self.conn.recv(HEADER_SZ))
        self.seg_idx = int(header[:HEADER_SEG_IDX_SZ].decode("utf8"))
        self.remaining_sz = int(header[HEADER_SEG_IDX_SZ:].decode("utf8"))

        logger.info(f"[App{self.server.app_idx}] Incoming video segment has "
                    f"seg_id {self.seg_idx}. Incoming video segment has "
                    f"{str(self.remaining_sz)} bytes")

    def _recv_video(self, path: str):
        # start recv video content
        out_file = open(path, "wb")
        while self.remaining_sz > 0:
            packet = self.conn.recv(PACKET_SZ)
            out_file.write(packet)
            self.remaining_sz -= len(packet)
        logger.info(
                f"[App{self.server.app_idx}] Received"
                f" all packets from segment {self.seg_idx}"
                )
        out_file.close()

    def run(self) -> None:

        self._recv_header()
        out_path = os.path.join(
            OUT_DIR, str(self.server.app_idx), f"encoded-{self.seg_idx}.avi")
        self._recv_video(out_path)

        # extract image from the video segment
        future = self.server.p_executor.submit(
                extract_img, vpath=out_path, seg_idx=self.seg_idx,
                app_idx=self.server.app_idx)
        while not future.done():
            pass
        logger.info(f"[App{self.server.app_idx}] Extracted all frames from"
                    f" segment {self.seg_idx}")

        # start inference loop
        img_idx = 1
        cur_img_path = os.path.join(
                OUT_DIR,
                str(self.server.app_idx),
                f"{self.seg_idx}-{img_idx}.png")
        frames: List[str] = []
        while os.path.isfile(cur_img_path):
            frames.append(cur_img_path)
            img_idx += 1
            cur_img_path = os.path.join(OUT_DIR, str(self.server.app_idx),
                                        f"{self.seg_idx}-{img_idx}.png")

        done_flag = threading.Event()
        self.server.detector.frames_queue_lock.acquire()
        self.server.detector.frames_queue.put((done_flag, frames))
        self.server.detector.frames_queue_lock.release()
        while not done_flag.is_set():
            pass

        # done inference, cleanup memory
        res: InferCache = InferCache(
                out_path, 0, self.server.detector.inference_queue.get())
        self.server.update_prev_iter(res)
        logger.info(f"[App{self.server.app_idx}] Finished"
                    f" inference for video segment {self.seg_idx}")
        self.conn.shutdown(socket.SHUT_RDWR)
        self.conn.close()
        self.server.job_done(self.seg_idx)
        return


def main():
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-sm", "--saved_model",
                        help="path to the tensorflow saved_model directory",
                        type=str,
                        default=os.environ["MODEL_PATH"])
    parser.add_argument("-th", "--threshold",
                        help="min score allowed for visualizing detection "
                             "boxes",
                        type=float,
                        default=0.3)
    parser.add_argument("-i", "--app_idx",
                        help="App ID",
                        type=int,
                        default=0)
    parser.add_argument("-c", "--client_ip",
                        help="Edge client ip address without port",
                        type=str,
                        required=True)
    args = parser.parse_args()
    app_idx = args.app_idx
    client_ip = args.client_ip

    control_port = 5000 + app_idx
    data_port = 10000 + app_idx
    logger.info("Loading saved model")
    detector = Detector(os.environ["MODEL_PATH"], args.threshold)
    logger.info("Object Detector initialized")

    server = Server(f"{client_ip}", True, data_port, detector, app_idx,
                    control_port)
    threading.Thread(target=grpc_serve, args=(server,)).start()
    time.sleep(5)
    server.checkin()
    server.serve()


if __name__ == "__main__":
    main()
