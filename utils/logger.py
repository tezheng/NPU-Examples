import logging

from colorama import Fore, Style


class ColorFormatter(logging.Formatter):
  COLORS = {
    logging.DEBUG: Style.DIM + Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Style.BRIGHT + Fore.RED,
  }

  RESET = Style.RESET_ALL

  def format(self, record):
    log_color = self.COLORS.get(record.levelno, self.RESET)
    message = super().format(record)
    return f"{log_color}{message}{self.RESET}"


def get_logger(name: str, level=logging.INFO):
  handler = logging.StreamHandler()
  handler.setLevel(level)
  handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))

  logger = logging.getLogger(name)
  logger.handlers = []
  logger.propagate = False
  logger.setLevel(level)
  logger.addHandler(handler)

  return logger
