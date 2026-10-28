# patch_windows.py
import sys

print("[Patch] Applying Windows compatibility...")

class TermiosCompat:
    TCIFLUSH = 0
    TCIOFLUSH = 1
    TCOFLUSH = 2
    TCSANOW = 0
    TCSADRAIN = 1
    TCSAFLUSH = 2
    
    @staticmethod
    def tcgetattr(fd): return None
    @staticmethod
    def tcsetattr(fd, when, attrs): pass
    @staticmethod
    def tcflush(fd, queue): pass
    @staticmethod
    def tcflow(fd, action): pass
    @staticmethod
    def tcsendbreak(fd, duration): pass
    @staticmethod
    def cfgetispeed(attrs): return 9600
    @staticmethod
    def cfgetospeed(attrs): return 9600
    @staticmethod
    def cfsetispeed(attrs, speed): pass
    @staticmethod
    def cfsetospeed(attrs, speed): pass
    @staticmethod
    def tcgetwinsize(fd): return (80, 24)

class TtyCompat:
    TCSAFLUSH = 2
    TCSADRAIN = 1
    TCSANOW = 0
    
    @staticmethod
    def setraw(fd, when=2): pass
    @staticmethod
    def setcbreak(fd, when=2): pass

# 强制注入
sys.modules['termios'] = TermiosCompat()
sys.modules['tty'] = TtyCompat()

print("[Patch] Complete!")