from concurrent.futures import ThreadPoolExecutor

try:
    from PySide6.QtCore import QRunnable, QThreadPool
except Exception:
    QRunnable = None
    QThreadPool = None


_FALLBACK_EXECUTOR = ThreadPoolExecutor(max_workers=8)


# 通用的 QRunnable 类
class SimpleRunnable(QRunnable if QRunnable is not None else object):
    def __init__(self, func, *args, **kwargs):
        if QRunnable is not None:
            super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            print(e)


# 通用的线程池运行函数
def run_in_threadpool(func, *args, **kwargs):
    if QThreadPool is not None:
        runnable = SimpleRunnable(func, *args, **kwargs)
        QThreadPool.globalInstance().start(runnable)
        return

    _FALLBACK_EXECUTOR.submit(func, *args, **kwargs)