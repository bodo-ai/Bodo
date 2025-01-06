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

  /**
   * Get the loggers that are enabled for the given level.
   *
   * @param level The Bodo verbose level.
   * @return The list of loggers that are enabled for the given level.
   */
  public static void toggleLoggers(int level) {
    if (level >= 1) {
      turnLoggerOn(VERBOSE_LEVEL_ONE_LOGGER);
    } else {
      turnLoggerOff(VERBOSE_LEVEL_ONE_LOGGER);
    }
    if (level >= 2) {
      turnLoggerOn(VERBOSE_LEVEL_TWO_LOGGER);
    } else {
      turnLoggerOff(VERBOSE_LEVEL_TWO_LOGGER);
    }
    if (level >= 3) {
      turnLoggerOn(VERBOSE_LEVEL_THREE_LOGGER);
    } else {
      turnLoggerOff(VERBOSE_LEVEL_THREE_LOGGER);
    }
  }

  static {
    // Disable loggers until turned on by Python.
    toggleLoggers(0);
  }
}
