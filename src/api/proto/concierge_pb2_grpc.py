import sys
import grpc
import logging

import api.proto.concierge_pb2 as concierge__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

# Set up global logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(name)-14s %(message)s',
                    handlers=[
                        logging.FileHandler("./concierge.log"),
                        logging.StreamHandler()
                    ])

# Apply global logger to the servicer and stub
servicer_logger = logging.getLogger("ConciergeServicer")
servicer_logger.setLevel(logging.DEBUG)

stub_logger = logging.getLogger("ConciergeStub")
stub_logger.setLevel(logging.DEBUG)

class ConciergeStub(object):
    """Client-side stub for making gRPC requests."""

    def __init__(self, channel):
        stub_logger.info("Initializing ConciergeStub with channel")
        self.CheckinApp = channel.unary_unary(
            '/Concierge/CheckinApp',
            request_serializer=concierge__pb2.AppInfo.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.RequestResource = channel.unary_unary(
            '/Concierge/RequestResource',
            request_serializer=concierge__pb2.ResRequest.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.ReadyToProfile = channel.unary_unary(
            '/Concierge/ReadyToProfile',
            request_serializer=concierge__pb2.PID.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.DoneProfile = channel.unary_unary(
            '/Concierge/DoneProfile',
            request_serializer=concierge__pb2.PID.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.SubmitGPUJob = channel.unary_unary(
            '/Concierge/SubmitGPUJob',
            request_serializer=concierge__pb2.AppInfo.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.DoneGPUJob = channel.unary_unary(
            '/Concierge/DoneGPUJob',
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.DoneInference = channel.unary_unary(
            '/Concierge/DoneInference',
            request_serializer=concierge__pb2.AppInfo.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )
        self.BaselineBW = channel.unary_unary(
            '/Concierge/BaselineBW',
            request_serializer=concierge__pb2.BaselineInfo.SerializeToString,
            response_deserializer=concierge__pb2.Status.FromString,
        )

    def call_checkin_app(self, request):
        stub_logger.info("Calling CheckinApp with request: %s", request)
        response = self.CheckinApp(request)
        stub_logger.info("Received CheckinApp response: %s", response)
        return response

    # Implement similar methods for other RPC calls with logging


class ConciergeServicer(object):
    """Server-side implementation of gRPC methods."""

    def CheckinApp(self, request, context):
        servicer_logger.info("Received CheckinApp request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("CheckinApp response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in CheckinApp: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def RequestResource(self, request, context):
        servicer_logger.info("Received RequestResource request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("RequestResource response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in RequestResource: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def ReadyToProfile(self, request, context):
        servicer_logger.info("Received ReadyToProfile request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("ReadyToProfile response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in ReadyToProfile: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def DoneProfile(self, request, context):
        servicer_logger.info("Received DoneProfile request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("DoneProfile response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in DoneProfile: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def SubmitGPUJob(self, request, context):
        servicer_logger.info("Received SubmitGPUJob request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("SubmitGPUJob response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in SubmitGPUJob: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def DoneGPUJob(self, request, context):
        servicer_logger.info("Received DoneGPUJob request")
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("DoneGPUJob response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in DoneGPUJob: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def DoneInference(self, request, context):
        servicer_logger.info("Received DoneInference request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("DoneInference response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in DoneInference: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def BaselineBW(self, request, context):
        servicer_logger.info("Received BaselineBW request: %s", request)
        try:
            response = concierge__pb2.Status(is_succeed=True)
            servicer_logger.info("BaselineBW response: %s", response)
            return response
        except Exception as e:
            servicer_logger.error("Error in BaselineBW: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise


def add_ConciergeServicer_to_server(servicer, server):
    servicer_logger.info("Adding ConciergeServicer to server")
    rpc_method_handlers = {
        'CheckinApp': grpc.unary_unary_rpc_method_handler(
            servicer.CheckinApp,
            request_deserializer=concierge__pb2.AppInfo.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'RequestResource': grpc.unary_unary_rpc_method_handler(
            servicer.RequestResource,
            request_deserializer=concierge__pb2.ResRequest.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'ReadyToProfile': grpc.unary_unary_rpc_method_handler(
            servicer.ReadyToProfile,
            request_deserializer=concierge__pb2.PID.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'DoneProfile': grpc.unary_unary_rpc_method_handler(
            servicer.DoneProfile,
            request_deserializer=concierge__pb2.PID.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'SubmitGPUJob': grpc.unary_unary_rpc_method_handler(
            servicer.SubmitGPUJob,
            request_deserializer=concierge__pb2.AppInfo.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'DoneGPUJob': grpc.unary_unary_rpc_method_handler(
            servicer.DoneGPUJob,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'DoneInference': grpc.unary_unary_rpc_method_handler(
            servicer.DoneInference,
            request_deserializer=concierge__pb2.AppInfo.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
        'BaselineBW': grpc.unary_unary_rpc_method_handler(
            servicer.BaselineBW,
            request_deserializer=concierge__pb2.BaselineInfo.FromString,
            response_serializer=concierge__pb2.Status.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'Concierge', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    servicer_logger.info("ConciergeServicer added to server")


# This class is part of an EXPERIMENTAL API.
class Concierge(object):
    """Experimental API for gRPC calls."""

    @staticmethod
    def CheckinApp(request, target, options=(), channel_credentials=None,
                   call_credentials=None, insecure=False, compression=None,
                   wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending CheckinApp request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/CheckinApp',
            concierge__pb2.AppInfo.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RequestResource(request, target, options=(), channel_credentials=None,
                        call_credentials=None, insecure=False, compression=None,
                        wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending RequestResource request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/RequestResource',
            concierge__pb2.ResRequest.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadyToProfile(request, target, options=(), channel_credentials=None,
                       call_credentials=None, insecure=False, compression=None,
                       wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending ReadyToProfile request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/ReadyToProfile',
            concierge__pb2.PID.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DoneProfile(request, target, options=(), channel_credentials=None,
                    call_credentials=None, insecure=False, compression=None,
                    wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending DoneProfile request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/DoneProfile',
            concierge__pb2.PID.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitGPUJob(request, target, options=(), channel_credentials=None,
                     call_credentials=None, insecure=False, compression=None,
                     wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending SubmitGPUJob request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/SubmitGPUJob',
            concierge__pb2.AppInfo.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DoneGPUJob(request, target, options=(), channel_credentials=None,
                   call_credentials=None, insecure=False, compression=None,
                   wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending DoneGPUJob request")
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/DoneGPUJob',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DoneInference(request, target, options=(), channel_credentials=None,
                      call_credentials=None, insecure=False, compression=None,
                      wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending DoneInference request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/DoneInference',
            concierge__pb2.AppInfo.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BaselineBW(request, target, options=(), channel_credentials=None,
                   call_credentials=None, insecure=False, compression=None,
                   wait_for_ready=None, timeout=None, metadata=None):
        stub_logger.info("Sending BaselineBW request: %s", request)
        return grpc.experimental.unary_unary(
            request, target, '/Concierge/BaselineBW',
            concierge__pb2.BaselineInfo.SerializeToString,
            concierge__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
