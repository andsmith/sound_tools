"""
Packing and unpacking wave datam where:
    Unpacked = numpy array of ints/floats
    packed = bytes

FROM:  http://trac.ffmpeg.org/wiki/audio%20types

 * DE alaw            PCM A-law
 DE f32be           PCM 32-bit floating-point big-endian
 DE f32le           PCM 32-bit floating-point little-endian
 DE f64be           PCM 64-bit floating-point big-endian
 DE f64le           PCM 64-bit floating-point little-endian
 * DE mulaw           PCM mu-law
 DE s16be           PCM signed 16-bit big-endian
 DE s16le           PCM signed 16-bit little-endian
 DE s24be           PCM signed 24-bit big-endian
 DE s24le           PCM signed 24-bit little-endian
 DE s32be           PCM signed 32-bit big-endian
 DE s32le           PCM signed 32-bit little-endian
 DE s8              PCM signed 8-bit
 DE u16be           PCM unsigned 16-bit big-endian
 DE u16le           PCM unsigned 16-bit little-endian
 DE u24be           PCM unsigned 24-bit big-endian
 DE u24le           PCM unsigned 24-bit little-endian
 DE u32be           PCM unsigned 32-bit big-endian
 DE u32le           PCM unsigned 32-bit little-endian
 DE u8              PCM unsigned 8-bit

 * unimplemented here
"""
import numpy as np
from construct import (Int8un, Int8sn, Int16ub, Int16ul, Int16sb, Int16sl, Int24ub, Int24ul, Int24sb, Int24sl,
                       Int32ub, Int32ul, Int32sb, Int32sl, Float32b, Float32l, Float64b, Float64l, Array, BytesInteger)

# For each of the above types, what kind of ndarray should the values be stored in?
NUMPY_TYPES_FOR_PCM_DATA_TYPE = {'s8': np.int8,
                                 's16le': np.int16,
                                 's16be': np.int16,
                                 's24le': np.int32,
                                 's24be': np.int32,
                                 's32le': np.int32,
                                 's32be': np.int32,
                                 'u8': np.uint8,
                                 'u16le': np.uint16,
                                 'u16be': np.uint16,
                                 'u24le': np.uint32,
                                 'u24be': np.uint32,
                                 'u32le': np.uint32,
                                 'u32be': np.uint32,
                                 'f32be': np.float32,
                                 'f32le': np.float32,
                                 'f64be': np.float64,
                                 'f64le': np.float64,
                                 }

# What is the data type for each PCM encoding string?
CONSTRUCT_TYPES_FOR_PCM_DATA_TYPE = {'s8': Int8sn,
                                     's16le': Int16sl,
                                     's16be': Int16sb,
                                     's24le': BytesInteger(3, swapped=True, signed=True),
                                     's24be': BytesInteger(3, swapped=False, signed=True),
                                     's32le': Int32sl,
                                     's32be': Int32sb,
                                     'u8': Int8un,
                                     'u16le': Int16ul,
                                     'u16be': Int16ub,
                                     'u24le': BytesInteger(3, swapped=True),
                                     'u24be': BytesInteger(3, swapped=False),
                                     'u32le': Int32ul,
                                     'u32be': Int32ub,
                                     'f32be': Float32b,
                                     'f32le': Float32l,
                                     'f64be': Float64b,
                                     'f64le': Float64l,
                                     }

CONSTRUCT_TYPES_NEEDING_LIST_CONV = ['s24le', 's24be', 'u24le', 'u24be']

SAMPLE_WIDTHS = {'s8': 1, 'u8': 1,
                 's16le': 2, 's16be': 2,
                 's24le': 3, 's24be': 3,
                 's32le': 4, 's32be': 4,
                 'u16le': 2, 'u16be': 2,
                 'u24le': 3, 'u24be': 3,
                 'u32le': 4, 'u32be': 4,
                 'f32be': 4, 'f32le': 4,
                 'f64be': 8, 'f64le': 8,
                 }

import logging
import subprocess
import re

# defaults for creating new sounds, TODO:  make a constructor param?
DEFAULT_ENCODING_FOR_SAMPLE_WIDTH = {1: 'u8',
                                     2: 's16le',
                                     3: 's24le',
                                     4: 's32le'}


def get_encoding_type(filename):
    """
    read wav file to get the audio encoding, which doesn't seem to be in wave.open().getparams(), etc.
    (how to convert bytes to numbers for each sample value)
    """
    cmd = ['ffprobe', '-hide_banner', filename]
    logging.info("Running:  %s" % (" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True)
    encoding = re.search('Audio: pcm_([^ ]*) ', str(result.stderr)).groups()[0]
    return encoding


def _bytes_to_numpy(b, encoding):
    """
    Convert PCM bytes to a numpy array
    :param b: bytes
    :param encoding: string, key from NUMPY_TYPES_FOR_PCM_DATA_TYPE
    :return: numpy array
    """
    n = int(len(b) / SAMPLE_WIDTHS[encoding])
    parser = Array(n, CONSTRUCT_TYPES_FOR_PCM_DATA_TYPE[encoding])
    values = parser.parse(b)

    return np.array(values)


def convert_from_bytes(data, encoding, n_channels):
    # figure out data type
    n_data = _bytes_to_numpy(data, encoding)

    # separate interleaved channel data
    n_data = [n_data[offset::n_channels] for offset in range(n_channels)]

    return n_data


def _numpy_to_bytes(a, encoding):
    """
    Convert a numpy array to PCM bytes
    :param a: array
    :param encoding: string, key from NUMPY_TYPES_FOR_PCM_DATA_TYPE
    :return: byte array
    """
    encoder = Array(a.size, CONSTRUCT_TYPES_FOR_PCM_DATA_TYPE[encoding])

    if encoding in CONSTRUCT_TYPES_NEEDING_LIST_CONV:
        byte_vals = encoder.build(a.tolist())  # bug in construct.BytesInteger?
    else:
        byte_vals = encoder.build(a)
    return byte_vals


def convert_to_bytes(chan_float_data, encoding):
    # interleave channel data
    n_chan = len(chan_float_data)
    data = np.zeros(n_chan * chan_float_data[0].size, dtype=NUMPY_TYPES_FOR_PCM_DATA_TYPE[encoding])
    for i_chan in range(n_chan):
        data[i_chan::n_chan] = chan_float_data[i_chan]

    return _numpy_to_bytes(data, encoding)
