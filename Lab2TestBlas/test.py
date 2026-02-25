import ctypes
import numpy as np
import traceback
import subprocess
import sys
import threading
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans  = 111
CblasTrans    = 112
CblasConjTrans= 113
CblasUpper    = 121
CblasLower    = 122
CblasNonUnit  = 131


class BlasL2Tester:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.types = {
            "s": (np.float32, ctypes.c_float),
            "d": (np.float64, ctypes.c_double),
            "c": (np.complex64, ctypes.c_float),
            "z": (np.complex128, ctypes.c_double),
        }

    def ptr(self, arr, ctype):
        return arr.ctypes.data_as(ctypes.POINTER(ctype))

    def base_vectors(self, dtype):
        x = np.array([1,2,3,4], dtype=dtype)
        y = np.array([4,3,2,1], dtype=dtype)
        return x, y

    def base_matrix(self, dtype):
        return np.eye(4, dtype=dtype)
    def run_threaded(self, prefix, test_name, num_threads=4, repeats=5):
        errors = []
        def worker():
            try:
                for _ in range(repeats):
                    getattr(self, f"test_{test_name}")(prefix)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
    def test_gemv(self, prefix):
        dtype, ctype = self.types[prefix]
        f = getattr(self.lib, f"cblas_{prefix}gemv")
        A = self.base_matrix(dtype)
        x, y = self.base_vectors(dtype)
        y_ref = A @ x

        if prefix in ["s","d"]:
            f(CblasRowMajor, CblasNoTrans, 4,4, ctype(1),
              self.ptr(A,ctype),4, self.ptr(x,ctype),1,
              ctype(0), self.ptr(y,ctype),1)
        else:
            a = np.array(1, dtype=dtype)
            b = np.array(0, dtype=dtype)
            f(CblasRowMajor, CblasNoTrans, 4,4,
              self.ptr(a,ctype), self.ptr(A,ctype),4,
              self.ptr(x,ctype),1, self.ptr(b,ctype), self.ptr(y,ctype),1)

        if not np.allclose(y, y_ref):
            raise AssertionError("Wrong result in gemv")

    def test_gbmv(self, prefix):
        dtype, ctype = self.types[prefix]
        f = getattr(self.lib, f"cblas_{prefix}gbmv")
        n = 4
        KL = KU = 0
        A_full = np.eye(n, dtype=dtype)
        A_band = np.zeros((n, KL+KU+1), dtype=dtype)
        for i in range(n):
            A_band[i, KU] = A_full[i, i]
        x, y = self.base_vectors(dtype)
        y_ref = A_full @ x

        if prefix in ["s","d"]:
            f(CblasRowMajor, CblasNoTrans,
              n, n, KL, KU,
              ctype(1), self.ptr(A_band,ctype), KL+KU+1,
              self.ptr(x,ctype),1, ctype(0), self.ptr(y,ctype),1)
        else:
            a = np.array(1, dtype=dtype)
            b = np.array(0, dtype=dtype)
            f(CblasRowMajor, CblasNoTrans,
              n, n, KL, KU,
              self.ptr(a,ctype), self.ptr(A_band,ctype), KL+KU+1,
              self.ptr(x,ctype),1, self.ptr(b,ctype), self.ptr(y,ctype),1)

        if not np.allclose(y, y_ref):
            raise AssertionError("Wrong result in gbmv")

    def test_trmv(self, prefix):
        dtype, ctype = self.types[prefix]
        f = getattr(self.lib, f"cblas_{prefix}trmv")
        A = self.base_matrix(dtype)
        x, _ = self.base_vectors(dtype)
        x_ref = A @ x
        f(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4,
          self.ptr(A,ctype),4, self.ptr(x,ctype),1)
        if not np.allclose(x, x_ref):
            raise AssertionError("Wrong result in trmv")

    def test_trsv(self, prefix):
        dtype, ctype = self.types[prefix]
        f = getattr(self.lib, f"cblas_{prefix}trsv")
        A = self.base_matrix(dtype)
        b, _ = self.base_vectors(dtype)
        x = b.copy()
        f(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 4,
          self.ptr(A,ctype),4, self.ptr(x,ctype),1)
        if not np.allclose(x, b):
            raise AssertionError("Wrong result in trsv")

    def test_ger(self, prefix):
        if prefix not in ["s","d"]:
            return
        dtype, ctype = self.types[prefix]
        f = getattr(self.lib, f"cblas_{prefix}ger")
        x, y = self.base_vectors(dtype)
        A = np.zeros((4,4), dtype=dtype)
        A_ref = np.outer(x, y)
        f(CblasRowMajor, 4,4, ctype(1), self.ptr(x,ctype),1, self.ptr(y,ctype),1, self.ptr(A,ctype),4)
        if not np.allclose(A, A_ref):
            raise AssertionError("Wrong result in ger")

    def test_symv_hemv(self, prefix):
        dtype, ctype = self.types[prefix]
        name = "symv" if prefix in ["s","d"] else "hemv"
        f = getattr(self.lib, f"cblas_{prefix}{name}")
        A = self.base_matrix(dtype)
        x, y = self.base_vectors(dtype)
        y_ref = A @ x
        if prefix in ["s","d"]:
            f(CblasRowMajor, CblasUpper, 4, ctype(1), self.ptr(A,ctype),4,
              self.ptr(x,ctype),1, ctype(0), self.ptr(y,ctype),1)
        else:
            a = np.array(1, dtype=dtype)
            b = np.array(0, dtype=dtype)
            f(CblasRowMajor, CblasUpper, 4,
              self.ptr(a,ctype), self.ptr(A,ctype),4, self.ptr(x,ctype),1,
              self.ptr(b,ctype), self.ptr(y,ctype),1)
        if not np.allclose(y, y_ref):
            raise AssertionError("Wrong result in symv/hemv")

    def _child_invoke(self, prefix, test_name, mode="one", threads=1):
        args = [sys.executable, __file__, "child", prefix, test_name, mode, str(threads)]
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_test_pair(self, prefix, test_name, multithread_count=8):
        cp = self._child_invoke(prefix, test_name, mode="one", threads=1)
        label_one = f"One Thread Test: {prefix}{test_name}"
        if cp.returncode == 0:
            print(Fore.GREEN + label_one + " PASSED" + Style.RESET_ALL)
        else:
            print(Fore.RED + label_one + " FAILED" + Style.RESET_ALL)
            if cp.stderr:
                print(cp.stderr.decode(), end="")
        cp2 = self._child_invoke(prefix, test_name, mode="multi", threads=multithread_count)
        label_multi = f"Multithread Test: {prefix}{test_name} [{multithread_count}T]"
        if cp2.returncode == 0:
            print(Fore.YELLOW + label_multi + " PASSED" + Style.RESET_ALL)
        else:
            print(Fore.RED + label_multi + " FAILED" + Style.RESET_ALL)
            if cp2.stderr:
                print(cp2.stderr.decode(), end="")

    def run_all(self):
        tests = ["gemv","gbmv","trmv","trsv","symv_hemv","ger"]
        for prefix in self.types:
            print("\n=== PREFIX {} ===".format(prefix.upper()))
            for t in tests:
                self.run_test_pair(prefix, t, multithread_count=8)
lib_path = "/home/rook/Desktop/OpenBLAS-develop/lib/libopenblas.so" #CHANGE TO YOUR PATH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def _child_main():
    if len(sys.argv) < 5:
        print("child usage: child <prefix> <test_name> <mode(one|multi)> <threads>", file=sys.stderr)
        sys.exit(2)
    prefix = sys.argv[2]
    test_name = sys.argv[3]
    mode = sys.argv[4]
    threads = int(sys.argv[5]) if len(sys.argv) >= 6 else 1

    tester = BlasL2Tester(lib_path)
    try:
        if mode == "one":
            getattr(tester, f"test_{test_name}")(prefix)
        else:
            tester.run_threaded(prefix, test_name, num_threads=threads, repeats=5)
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if len(sys.argv) >= 2 and sys.argv[1] == "child":
    _child_main()
else:
    tester = BlasL2Tester(lib_path)
    tester.run_all()
