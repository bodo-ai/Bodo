cimport pyarrow.lib
from libcpp.memory cimport shared_ptr
from pyarrow._dataset cimport Scanner
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_dataset cimport CScanner


cdef public class ScannerBodo(Scanner) [object c_ScannerBodo, type c_ScannerBodo_t]:
    def __init__(self):
        Scanner.__init__(self)


cdef public shared_ptr[CScanner] pyarrow_unwrap_scanner_bodo(ScannerBodo scanner):
    return scanner.unwrap()
