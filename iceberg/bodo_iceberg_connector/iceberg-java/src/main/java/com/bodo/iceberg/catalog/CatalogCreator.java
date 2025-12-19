package com.bodo.iceberg.catalog;

import com.bodo.iceberg.Triple;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hc.core5.http.NameValuePair;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;

/** Iceberg Catalog Connector and Communicator */
public class CatalogCreator {
  public static Triple<Configuration, Map<String, String>, URIBuilder> prepareInput(
      String connStr, String catalogType, String coreSitePath) throws URISyntaxException {
    // Extract Parameters from URI
    // TODO: Just get from Python
    URIBuilder uriBuilder = new URIBuilder(connStr);
    Map<String, String> params =
        uriBuilder.getQueryParams().stream()
            .collect(Collectors.toMap(NameValuePair::getName, NameValuePair::getValue));
    params.remove("type");

    // Create Configuration
    // Additional parameters like Iceberg-specific ones should be ignored
    // Since the conf is reused by multiple objects, like Hive and Hadoop ones
    // TODO: Spark does something similar, but I believe they do some filtering. What is it?
    boolean loadDefaults = Objects.equals(coreSitePath, "");
    Configuration conf = new Configuration(loadDefaults);
    // Core site path is specified
    if (!loadDefaults) {
      conf.addResource(new Path(coreSitePath));
    }
    for (Map.Entry<String, String> entry : params.entrySet()) {
      conf.set(entry.getKey(), entry.getValue());
    }

    CatalogCreator.configureAzureAuth(conf);

    // Catalog URI (without parameters)
    String uriStr = uriBuilder.removeQuery().build().toString();
    params.put(CatalogProperties.URI, uriStr);

    return new Triple<>(conf, params, uriBuilder);
  }

  public static Catalog create(String connStr, String catalogType, String coreSitePath)
      throws URISyntaxException {
    // Create Catalog
    final Catalog catalog;

    // S3Tables doesn't use a URI
    if (connStr.startsWith("arn:aws:s3tables") && catalogType.equals("s3tables")) {
      catalog = S3TablesBuilder.create(connStr);
      return catalog;
    }

    // Avoid URI parsing with Windows paths like "C:\..."
    if (catalogType.equalsIgnoreCase("hadoop")
        && System.getProperty("os.name").toLowerCase().contains("win")) {
      Configuration conf = new Configuration(true);
      Map<String, String> params = new HashMap<>();
      return HadoopBuilder.create(connStr, conf, params);
    }

    var out = prepareInput(connStr, catalogType, coreSitePath);
    Configuration conf = out.getFirst();
    Map<String, String> params = out.getSecond();
    URIBuilder uriBuilder = out.getThird();

    switch (catalogType.toLowerCase()) {
      case "hive":
        catalog = ThriftBuilder.create(conf, params);
        break;
      case "glue":
        catalog = GlueBuilder.create(conf, params);
        break;
      case "hadoop":
        catalog =
            HadoopBuilder.create(
                uriBuilder.removeQuery().setScheme("").build().toString(), conf, params);
        break;
      case "hadoop-s3":
      case "hadoop-abfs":
        catalog = HadoopBuilder.create(uriBuilder.removeQuery().build().toString(), conf, params);
        break;
      case "rest":
        catalog = RESTBuilder.create(conf, params);
        break;
      default:
        throw new UnsupportedOperationException("Should never occur. Captured in Python");
    }

    return catalog;
  }

  /**
   * Configure Hadoop Azure authentication for the given Configuration object.
   *
   * @param conf Configuration object to configure Azure authentication for
   */
  private static void configureAzureAuth(Configuration conf) {
    String accountName = System.getenv("AZURE_STORAGE_ACCOUNT_NAME");
    String accountKey = System.getenv("AZURE_STORAGE_ACCOUNT_KEY");
    if (accountName != null && accountKey != null) {
      conf.set("fs.azure.account.key." + accountName + ".dfs.core.windows.net", accountKey);
    } else if (System.getenv("BODO_PLATFORM_CLOUD_PROVIDER") != null
        && System.getenv("BODO_PLATFORM_CLOUD_PROVIDER").equals("AZURE")) {
      // We're on the platform in an Azure workspace, try to use the identity
      conf.set("fs.azure.account.auth.type", "OAuth");
      conf.set(
          "fs.azure.account.oauth.provider.type",
          "org.apache.hadoop.fs.azurebfs.oauth2.MsiTokenProvider");
      conf.set("fs.azure.account.oauth2.msi.tenant", "");
      conf.set("fs.azure.account.oauth2.client.id", "");
      conf.set("fs.azure.account.oauth2.msi.endpoint", "");
    }
  }
}
