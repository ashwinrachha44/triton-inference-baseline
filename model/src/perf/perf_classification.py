import argparse
import logging
import json
import os
import cProfile, pstats
import numpy as np
import tritonclient.http

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, help='Test file')
    parser.add_argument('--model_endpoint', type=str, help='Model endpoint')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--model_version', type=str, help='Model version')
    parser.add_argument('--output_dir', type=str, help='Output dir')
    parser.add_argument('--num_cases', type=int, help='Number of test cases', default=-1)
    parser.add_argument('--debug', type=bool, help='Enable debugging', default=False)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    logger.info(f'create triton client for endpoint {args.model_endpoint} - model {args.model_name} - version {args.model_version}')
    triton_client = tritonclient.http.InferenceServerClient(url=args.model_endpoint, verbose=False)
    assert triton_client.is_model_ready(model_name=args.model_name, model_version=args.model_version), f'endpoint {args.model_endpoint} - model {args.model_name} - version {args.model_version} is not ready'

    model_metadata = triton_client.get_model_metadata(model_name=args.model_name, model_version=args.model_version)
    model_config = triton_client.get_model_config(model_name=args.model_name, model_version=args.model_version)
    logger.info(f'model metadata: {model_metadata}')
    logger.info(f'model config: {model_config}')

    test_cases = []
    with open(args.test_file, 'r', encoding='utf-8') as in_file:
        lines = in_file.read().splitlines()
        for line in lines:
            d = json.loads(line)
            test_cases.append(d)

    num_cases = len(test_cases)
    if args.num_cases > 0:
        num_cases = min(args.num_cases, num_cases)
        test_cases = test_cases[:num_cases]
    logger.info(f'load {num_cases} / {len(test_cases)} test cases from {args.test_file}')

    def infer(examples):
        input = tritonclient.http.InferInput(name='text', shape=(1,), datatype='BYTES')
        output = tritonclient.http.InferRequestedOutput(name='predict', binary_data=False)

        preds = []
        for i, d in enumerate(examples):
            input.set_data_from_numpy(np.array([d['text']], dtype=object))
            results = triton_client.infer(model_name=args.model_name, model_version=args.model_version, inputs=[input], outputs=[output])
            d['pred'] = results.as_numpy('predict').tolist()[0]
            if (i + 1) % 100 == 0 and args.debug:
                logger.info(f'infer {i+1} test cases')
            preds.append(d)

        return preds

    profiler = cProfile.Profile()
    profiler.enable()
    test_preds = infer(test_cases)
    profiler.disable()
    perf_file = os.path.join(args.output_dir, 'perf_stats.log')
    logger.info(f'save perf logs to {perf_file}')
    stats = pstats.Stats(profiler, stream=open(perf_file, 'w', encoding='utf-8')).sort_stats('tottime')
    stats.print_stats()

    predict_file = os.path.join(args.output_dir, 'predict_results.json')
    logger.info(f'save {num_cases} predict examples to {predict_file}')
    with open(predict_file, 'w', encoding='utf-8') as out_file:
        for d in test_preds:
            line = json.dumps(d)
            out_file.write(f'{line}\n')


if __name__ == "__main__":
    main()
