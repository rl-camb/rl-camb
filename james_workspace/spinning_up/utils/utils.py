import sys
import argparse

class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)


def smooth_over(list_to_smooth, smooth_last):
    smoothed = [list_to_smooth[0]]
    for i in range(1, len(list_to_smooth)+1):
        if i < smooth_last:
            smoothed.append(
                sum(list_to_smooth[:i]) / len(list_to_smooth[:i]))
        else:
            assert smooth_last == len(list_to_smooth[i-smooth_last:i])
            smoothed.append(
                sum(list_to_smooth[i-smooth_last:i]) / smooth_last
                )
    return smoothed