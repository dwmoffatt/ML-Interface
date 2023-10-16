"""
Machine Learning Command Line Interface
"""
import logging


if __name__ in ("__main__", "__builtin__"):

  logging.basicConfig(
      filename="Interface.log",
      format="[%(asctime)s][%(levelname)s] - %(message)s",
      filemode="w",
      level=logging.INFO,
  )
