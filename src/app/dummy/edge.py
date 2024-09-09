import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import subprocess
from threading import Lock, Thread
from time import sleep
import time
from typing import Dict, List

import app.proto.edge_pb2 as pb2
from app.proto.edge_pb2 import google_dot_protobuf_dot_empty__pb2 as empty
import app.proto.edge_pb2_grpc as pb2_grpc
import grpc


ENCODED_PATH = os.path.join(
        Path(Path(__file__).parent.resolve()).parents[1], "edge-res")
IN_PATH = os.path.join(ENCODED_PATH, "traffic.mp4")
PACKET_SZ = 4096

try:
    IF = os.environ["IF"] + "\0"
except KeyError:
    IF = "eno1" + "\0"


def get_video_length(path: str) -> float:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", path],
                         capture_output=True, text=True)
    try:
        return float(res.stdout)
    except ValueError:
        return 0


class Edge(pb2_grpc.EdgeServicer):
    def __init__(self, addr: tuple, app_idx: int):
        self.server_addr = addr
        self.app_idx = app_idx
        self.bitrate = -1
        self.jobs_start_time: Dict[int, int] = {}
        self.elapsed_times: List[int] = []
        self.fd_lock = Lock()

    def UpdateBitrate(self, request, _):
        print("UpdateBitrate called", flush=True)
        self.bitrate = int(request.bitrate / 8)
        print(f"Set bitrate to {self.bitrate}", flush=True)
        return pb2.IsUpdated(updated=True)

    def NotifyJobDone(self, request, _):
        seg_idx: int = request.seg_idx
        elapsed_time_ns = time.perf_counter_ns()\
            - self.jobs_start_time[seg_idx]
        self.elapsed_times.append(elapsed_time_ns)
        self.fd_lock.acquire()
        latency_fd = open(f"/tmp/client{self.app_idx}", "a+")
        latency_fd.write(f"{elapsed_time_ns}\n")
        print(f"{seg_idx} {elapsed_time_ns}", flush=True)
        latency_fd.close()
        self.fd_lock.release()
        return empty.Empty()

    def cut_compress_v(self, file_path: str, vrate: int, st: int,
                       duration: int, seg: int, bitrate: int) -> tuple:

        print(f"Encode at {bitrate}KBps", flush=True)
        out_dir = os.path.join(ENCODED_PATH, str(self.app_idx))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_filename = os.path.join(ENCODED_PATH, str(self.app_idx),
                                    f"{seg}.avi")
        cmd = ["ffmpeg", "-y", "-ss", str(st), "-t", str(duration),
               "-i", file_path, "-c:v", "libx264", "-an", "-r", str(vrate)]
        if bitrate > 0:
            cmd.append("-b:v")
            cmd.append(f"{bitrate}k")
        cmd.append(out_filename)

        compress_res = subprocess.run(cmd, capture_output=True, text=True)
        size = 0
        if compress_res.returncode != 0:
            # Encoding failed
            print("ENCODING FAILED", compress_res.stdout, compress_res.stderr)
            exit(-1)
        else:
            size = os.path.getsize(out_filename)

        return out_filename, size

    def send_video(self, path: str, seg_len: int, vrate: int) -> None:
        cur_seg = 0
        st = 0

        trans_end = False
        while not trans_end:

            skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # skt.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
            #                IF.encode("utf8"))
            skt.connect(self.server_addr)
            while self.bitrate < 0:
                pass

            compress_st = time.time()
            out_filename, remain_sz = self.cut_compress_v(
                    path, vrate, st, seg_len, cur_seg, self.bitrate)
            print(f"Compress Time: {time.time() - compress_st}")
            # header
            # fixed 12 bytes for the video segment identification
            # fixed 16 bytes for the size of the video segment
            cur_seg_byte = ("%12s" % str(cur_seg)).encode("utf8")
            total_sz_byte = ("%16s" % str(remain_sz)).encode("utf8")
            skt.send(cur_seg_byte + total_sz_byte)

            # payload
            in_f = open(out_filename, mode="rb")
            packet = in_f.read(remain_sz)
            skt.sendall(packet)
            skt.shutdown(socket.SHUT_RDWR)
            skt.close()

            # segment finished, do cleanup
            self.jobs_start_time[cur_seg] = time.perf_counter_ns()
            print(f"Edge has sent all packets in segment {cur_seg}",
                  flush=True)
            in_f.close()
            cur_seg += 1
            st += seg_len
            trans_end = get_video_length(out_filename) < seg_len
            if (cur_seg == 1):
                sleep(10)
            sleep(seg_len + 4)


def serve(servicer: Edge, control_port):
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_EdgeServicer_to_server(servicer, server)
    server.add_insecure_port(control_port)
    server.start()
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="server address",
                        default="localhost", type=str)
    parser.add_argument("-i", "--app_idx",
                        help="edge device index in the cluster",
                        type=int, required=True)
    parser.add_argument("--seg_len",
                        help="the length of the segment sent to the server "
                             "each time",
                        type=int, default=2)
    parser.add_argument("--vpath", help="input video path",
                        type=str, default=IN_PATH)

    args = parser.parse_args()
    port = 10000 + args.app_idx
    client = Edge(addr=(args.host, port), app_idx=args.app_idx)
    control_port = f"0.0.0.0:{20000 + args.app_idx}"
    Thread(target=serve, args=(client, control_port)).start()
    sleep(5)
    client.send_video(args.vpath, args.seg_len, 5)


if __name__ == "__main__":
    main()
