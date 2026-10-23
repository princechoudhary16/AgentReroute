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


class InvalidSynthesis(SlowTransport):
    def complete(self, stage, payload):
        answer = super().complete(stage, payload)
        return {'constraints': [{'invalid': 'constraint'}]} if stage.startswith('perspective_') else answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    with TemporaryDirectory(prefix='comparison-browser-') as output:
        service = ComparisonService(output, provider_factory=lambda model: InvalidSynthesis() if model == 'invalid-synthesis' else SlowTransport(),
                                    model_catalog=lambda: ['other-model', 'fixture-model', 'invalid-synthesis'])
        server = make_server(port=args.port, delay=0, comparison_service=service)
        print(f'http://127.0.0.1:{server.server_address[1]}', flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
