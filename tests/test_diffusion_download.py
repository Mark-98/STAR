import io
import unittest
from unittest.mock import patch
import zipfile
from scripts.download_diffusion_data import RangeReader


class RangeDownloadTests(unittest.TestCase):
    def setUp(self):
        self.payload = b'0123456789abcdefghijklmnopqrstuvwxyz'

    def response(self, request, **kwargs):
        first, last = map(int, request.headers['Range'][6:].split('-'))
        result = io.BytesIO(self.payload[first:last + 1])
        result.status = 206
        result.headers = {'Content-Range': f'bytes {first}-{last}/{len(self.payload)}', 'ETag': 'fixed'}
        return result

    def test_seek_read_and_eof(self):
        with patch('urllib.request.urlopen', side_effect=self.response), RangeReader('https://example.org/a.zip', 7) as reader:
            self.assertEqual(reader.read(10), self.payload[:10])
            reader.seek(-4, 2)
            self.assertEqual(reader.read(), self.payload[-4:])
            self.assertEqual(reader.read(1), b'')
            reader.seek(2)
            self.assertEqual(reader.read(3), b'234')

    def test_zip_member_streaming(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
            stream.writestr('can/ph/image_abs.hdf5', b'demonstrations' * 100)
        self.payload = archive.getvalue()
        with patch('urllib.request.urlopen', side_effect=self.response), RangeReader('https://example.org/a.zip', 19) as reader:
            with zipfile.ZipFile(reader) as stream:
                self.assertEqual(stream.read('can/ph/image_abs.hdf5'), b'demonstrations' * 100)

    def test_server_ignoring_ranges_is_rejected_without_reading_body(self):
        response = io.BytesIO(b'not a full download')
        response.status, response.headers = 200, {}
        with patch('urllib.request.urlopen', return_value=response), self.assertRaises(RuntimeError):
            RangeReader('https://example.org/a.zip')

    def test_wrong_range_is_rejected(self):
        response = io.BytesIO(b'x')
        response.status, response.headers = 206, {'Content-Range': 'bytes 1-1/100'}
        with patch('urllib.request.urlopen', return_value=response), self.assertRaises(RuntimeError):
            RangeReader('https://example.org/a.zip')
