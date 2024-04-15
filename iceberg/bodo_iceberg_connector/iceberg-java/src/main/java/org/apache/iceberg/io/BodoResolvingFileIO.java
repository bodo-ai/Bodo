package org.apache.iceberg.io;

/**
 * ResolvingFileIO class that converts wasb:// and wasbs:// paths to abfs:// and abfss:// paths.
 * This is used to simplify the core-site configuration.
 */
public class BodoResolvingFileIO extends ResolvingFileIO {
  public BodoResolvingFileIO() {
    super();
  }

  private String convertWasbToAbfs(String location) {
    if (location.startsWith("wasbs://") || location.startsWith("wasb://")) {
      String optimizedLocation = location;
      if (location.startsWith("wasbs://")) {
        optimizedLocation =
            optimizedLocation
                .replace("wasbs://", "abfss://")
                .replace("blob.core.windows.net", "dfs.core.windows.net");
      } else if (location.startsWith("wasb://")) {
        optimizedLocation =
            optimizedLocation
                .replace("wasb://", "abfs://")
                .replace("blob.core.windows.net", "dfs.core.windows.net");
      }
      return optimizedLocation;
    }
    return location;
  }

  @Override
  public InputFile newInputFile(String location) {
    return super.newInputFile(convertWasbToAbfs(location));
  }

  @Override
  public InputFile newInputFile(String location, long length) {
    return super.newInputFile(convertWasbToAbfs(location), length);
  }

  @Override
  public OutputFile newOutputFile(String location) {
    return super.newOutputFile(convertWasbToAbfs(location));
  }

  @Override
  public void deleteFile(String location) {
    super.deleteFile(convertWasbToAbfs(location));
  }

  @Override
  public Iterable<FileInfo> listPrefix(String prefix) {
    return super.listPrefix(prefix);
  }

  @Override
  public void deletePrefix(String prefix) {
    super.deletePrefix(convertWasbToAbfs(prefix));
  }
}
