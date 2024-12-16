package com.bodosql.calcite.application;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Defines loggers that will export debugging/error messages to Python. The loggers are defined by
 * Bodo verbose level, but Python is responsible for checking the verbose level to decide upon
 * displaying the logged information.
 */
public class PythonLoggers {
  public static final Logger VERBOSE_LEVEL_ONE_LOGGER = Logger.getLogger("VERBOSE_LEVEL_ONE");
  public static final Logger VERBOSE_LEVEL_TWO_LOGGER = Logger.getLogger("VERBOSE_LEVEL_TWO");

  public static final Logger VERBOSE_LEVEL_THREE_LOGGER = Logger.getLogger("VERBOSE_LEVEL_THREE");

  // Pattern copied from Py4j
  public static void turnLoggerOn(Logger logger) {
    logger.setLevel(Level.INFO);
  }

  public static void turnLoggerOff(Logger logger) {
    logger.setLevel(Level.OFF);
  }

  static {
    // Disable loggers until turned on by Python.
    turnLoggerOff(VERBOSE_LEVEL_ONE_LOGGER);
    turnLoggerOff(VERBOSE_LEVEL_TWO_LOGGER);
    turnLoggerOff(VERBOSE_LEVEL_THREE_LOGGER);
  }
}
