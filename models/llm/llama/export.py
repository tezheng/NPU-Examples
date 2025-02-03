import sys
from dataclasses import asdict

from .convert_onnx import (
  LlamaWithKVCache,
  CalibDataGenerator,
  ConversionConfig,
  ConvertONNX,
)
from .quant_qdq import QuantizationConfig, quant
from .util import logger, parse_args


ground_truth = [{
    'input': '''<|start_header_id|>system<|end_header_id|>
You are a space exploration history expert.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Who is the first female astronaut to walk on the moon?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>''',
    'output': 'The first female astronaut to walk on the moon was Valentina \
               Tereshkova, a Soviet cosmonaut. She flew aboard the Vostok 6 \
               spacecraft on June 16, 1963, and spent almost three days in \
               space, becoming the first woman to journey into outer space.',
   }, {
    'input': '''<|start_header_id|>system<|end_header_id|>
You are a American history expert.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Who is the first presedent of United States?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>''',
    'output': 'The first President of the United States was George Washington. \
               He was inaugurated on April 30, 1789, and served two terms in \
               office until March 4, 1797.',
   }]


def export():
  args = parse_args()

  model_name = args.model_name
  logger.info(f'Exporting model {model_name}')

  model_dir = args.model_dir
  model_dir.mkdir(parents=True, exist_ok=True)
  data_dir = args.data_dir
  data_dir.mkdir(parents=True, exist_ok=True)

  print(f"Data dir: {data_dir}")

  conv_cfg = ConversionConfig(
    skip_prefill=args.skip_prefill,
    skip_decode=args.skip_decode,
    max_samples=args.max_samples,
  )

  quant_cfg = QuantizationConfig(
    node_optimization=args.node_optimization,
    skip_pre_process=args.skip_pre_process,
  )

  if args.inference:
    logger.warning('Running inference only, ignore other operation flags...\n')
    model = LlamaWithKVCache(model_name, use_streaming=args.use_streaming)
    if args.prompt:
      logger.info(f"Prompt:\n{args.prompt}")
      model.run(args.prompt)
    else:
      for item in ground_truth:
        logger.info(f"Prompt:\n{item['input']}")
        model.run(item['input'])
    sys.exit(0)

  if args.gen_calib_data or args.all:
    logger.info('Generating calibration data...\n')
    calib_data_gen = CalibDataGenerator(
      model_name=model_name, **asdict(conv_cfg))
    for item in ground_truth:
      logger.info(f"Prompt:\n{item['input']}")
      calib_data_gen.run(item['input'])
    calib_data_gen.save_data(data_dir=data_dir)

  if args.convert_onnx or args.all:
    logger.info('Converting to ONNX...\n')
    converter = ConvertONNX(model_name=model_name, **asdict(conv_cfg))
    logger.debug("Model Summary:")
    logger.debug(converter._model)
    converter.run('Hello world!')
    converter.export(model_dir)

  if args.quantize_qdq or args.all:
    if not args.skip_prefill:
      logger.info('Quantizing prefill model...\n')
      quant(
        model_path=model_dir / 'prefill.onnx',
        data_path=data_dir / 'prefill.npz',
        output_dir=model_dir,
        config=quant_cfg,
      )

    if not args.skip_decode:
      logger.info('Quantizing decode model...\n')
      quant(
        model_path=model_dir / 'decode.onnx',
        data_path=data_dir / 'decode.npz',
        output_dir=model_dir,
        config=quant_cfg,
      )
