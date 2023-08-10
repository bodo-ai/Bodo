import os

if __name__ == "__main__":
    assert "CLASSPATH" in os.environ
    # Verify that that jar is already in the CLASSPATH (even w/o
    # importing the package). Since we copy the jar should to the HADOOP
    # directory, it should be added to the CLASSPATH as part
    # of .bashrc
    assert "bodo-azurefs-sas-token-provider.jar" in os.environ["CLASSPATH"]
