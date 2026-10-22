"""Isolated model-transport fixture for browser tests; never used by the real demo server."""
import argparse
import time
from tempfile import TemporaryDirectory

from demo.comparison import ComparisonService
from demo.visualizer import make_server
from demo.tests.test_comparison import TransportFixture


class SlowTransport(TransportFixture):
    def complete(self, stage, payload):
        time.sleep(.07)
        return super().complete(stage, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    with TemporaryDirectory(prefix='comparison-browser-') as output:
        service = ComparisonService(output, provider_factory=lambda _: SlowTransport(),
                                    model_catalog=lambda: ['other-model', 'fixture-model'])
        server = make_server(port=args.port, delay=0, comparison_service=service)
        print(f'http://127.0.0.1:{server.server_address[1]}', flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
