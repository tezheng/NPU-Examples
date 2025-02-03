import sys
from dataclasses import asdict

from .model import Qwen2WithKVCache
from .gen_calib_data import CalibDataGenerator, CalibConfig
from .convert_onnx import ConvertONNX
from .quant_qdq import QuantizationConfig, quant
from .util import logger, parse_args


ground_truth = [{
    'input': 'Who is the first astronaut walking on the moon?',
    'output': 'The first female astronaut to walk on the moon was Valentina \
               Tereshkova, a Soviet cosmonaut. She flew aboard the Vostok 6 \
               spacecraft on June 16, 1963, and spent almost three days in \
               space, becoming the first woman to journey into outer space.',
   }, {
    'input': 'Who is the first president of United States?',
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

  conv_cfg = CalibConfig(
    max_samples=args.max_samples,
    skip_prefill=args.skip_prefill,
    skip_decode=args.skip_decode,
  )

  quant_cfg = QuantizationConfig(
    node_optimization=args.node_optimization,
    skip_pre_process=args.skip_pre_process,
  )

  if args.inference:
    logger.warning('Running inference only, ignore other operation flags...\n')
    model = Qwen2WithKVCache(model_name, use_streaming=args.use_streaming)
    if args.prompt:
      # logger.info(f"Prompt:\n{args.prompt}")
      _, token_count = model.run(args.prompt)
      logger.info(f"Generated token count: {token_count}")
    else:
      for item in ground_truth:
        # logger.info(f"Prompt:\n{item['input']}")
        _, token_count = model.run(item['input'])
        logger.info(f"Generated token count: {token_count}")
    sys.exit(0)

  if args.gen_calib_data or args.all:
    logger.info('Generating calibration data...\n')
    prompt = "Hello, this is Sam speaking, it is so nice to meet you! Let me give you a simple puzzle, see whether you can solve it and then find out how smart you are, would you? Please bear with me, here is the puzzle: imagine you are in a room with three light switches. Each switch controls one of three light bulbs in another room. You cannot see the bulbs from where the switches are. You can flip the switches as many times as you want, but you can only enter the room with the bulbs once. How can you determine which switch controls which bulb?"
    calib_data_gen = CalibDataGenerator(model_name, **asdict(conv_cfg))
    calib_data_gen.run(prompt)
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
