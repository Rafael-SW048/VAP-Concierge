# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/edge.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/edge.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10proto/edge.proto\x1a\x1bgoogle/protobuf/empty.proto\"\x1a\n\x07\x42itrate\x12\x0f\n\x07\x62itrate\x18\x01 \x01(\x05\"\x1c\n\tIsUpdated\x12\x0f\n\x07updated\x18\x01 \x01(\x08\"*\n\tIsJobDone\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0f\n\x07seg_idx\x18\x02 \x01(\x05\x32\x66\n\x04\x45\x64ge\x12\'\n\rUpdateBitrate\x12\x08.Bitrate\x1a\n.IsUpdated\"\x00\x12\x35\n\rNotifyJobDone\x12\n.IsJobDone\x1a\x16.google.protobuf.Empty\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,])




_BITRATE = _descriptor.Descriptor(
  name='Bitrate',
  full_name='Bitrate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='bitrate', full_name='Bitrate.bitrate', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=75,
)


_ISUPDATED = _descriptor.Descriptor(
  name='IsUpdated',
  full_name='IsUpdated',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='updated', full_name='IsUpdated.updated', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=77,
  serialized_end=105,
)


_ISJOBDONE = _descriptor.Descriptor(
  name='IsJobDone',
  full_name='IsJobDone',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='done', full_name='IsJobDone.done', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='seg_idx', full_name='IsJobDone.seg_idx', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=149,
)

DESCRIPTOR.message_types_by_name['Bitrate'] = _BITRATE
DESCRIPTOR.message_types_by_name['IsUpdated'] = _ISUPDATED
DESCRIPTOR.message_types_by_name['IsJobDone'] = _ISJOBDONE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Bitrate = _reflection.GeneratedProtocolMessageType('Bitrate', (_message.Message,), {
  'DESCRIPTOR' : _BITRATE,
  '__module__' : 'proto.edge_pb2'
  # @@protoc_insertion_point(class_scope:Bitrate)
  })
_sym_db.RegisterMessage(Bitrate)

IsUpdated = _reflection.GeneratedProtocolMessageType('IsUpdated', (_message.Message,), {
  'DESCRIPTOR' : _ISUPDATED,
  '__module__' : 'proto.edge_pb2'
  # @@protoc_insertion_point(class_scope:IsUpdated)
  })
_sym_db.RegisterMessage(IsUpdated)

IsJobDone = _reflection.GeneratedProtocolMessageType('IsJobDone', (_message.Message,), {
  'DESCRIPTOR' : _ISJOBDONE,
  '__module__' : 'proto.edge_pb2'
  # @@protoc_insertion_point(class_scope:IsJobDone)
  })
_sym_db.RegisterMessage(IsJobDone)



_EDGE = _descriptor.ServiceDescriptor(
  name='Edge',
  full_name='Edge',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=151,
  serialized_end=253,
  methods=[
  _descriptor.MethodDescriptor(
    name='UpdateBitrate',
    full_name='Edge.UpdateBitrate',
    index=0,
    containing_service=None,
    input_type=_BITRATE,
    output_type=_ISUPDATED,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='NotifyJobDone',
    full_name='Edge.NotifyJobDone',
    index=1,
    containing_service=None,
    input_type=_ISJOBDONE,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_EDGE)

DESCRIPTOR.services_by_name['Edge'] = _EDGE

# @@protoc_insertion_point(module_scope)
