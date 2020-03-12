#!/usr/bin/env python
# encoding: utf-8

import collections
import argparse

class Op:
    def __init__(self):
        self.type = ''
        self.device = ''
        self.name = ''
        self.implementation = ''
        self.propagation = ''
        self.iodata = ''
        self.aux = ''
        self.reserved = ''
        self.benchdnn = ''
        self.time = ''

class DNNL_Verbose:
    def __init__(self, args):
        self.args = args
        self.version = ''
        self.cpu_runtime = ''
        self.cpu_isa = ''
        self.gpu_runtime = ''
        self.gpu_engine = ''
        self.ops = []

    def load(self):
        try:
            with open(self.args.logfile) as f:
                for line in f:
                    if line.startswith('mkldnn_verbose'):
                        line = line.replace('\r', '')
                        line = line.replace('\n', '')
                        c = line.split(',')
                        if c[1] == 'info':
                            if c[2].startswith('Intel'):
                                self.version = c[2]
                            if c[2].startswith('Detected ISA'):
                                self.cpu_isa = ','.join(c[2:]).replace('Detected ISA is ', '')
                        else:
                            op = Op()
                            op.type = c[1]
                            op.device = 'cpu'
                            op.name = c[2]
                            op.implementation = c[3]
                            op.propagation = c[4]
                            op.iodata = c[5]
                            op.aux = c[6]
                            op.reserved = ''
                            op.benchdnn = c[7]
                            op.time = c[8]
                            self.ops.append(op)
                    if line.startswith('dnnl_verbose,'):
                        line = line.replace('\r', '')
                        line = line.replace('\n', '')
                        c = line.split(',')
                        if c[1] == 'info':
                            if len(c) == 3:
                                self.version = c[2]
                            else:
                                if c[2] == 'cpu':
                                    if c[3].startswith('runtime:'):
                                        self.cpu_runtime = c[3].split(':')[1]
                                    if c[3].startswith('isa:'):
                                        self.cpu_isa = c[3].split(':')[1]
                                if c[2] == 'gpu':
                                    if c[3].startswith('runtime:'):
                                        self.gpu_runtime = c[3].split(':')[1]
                                    if c[3] == 'engine':
                                        self.gpu_engine = c[5].split(':')[1]
                        else:
                            op = Op()
                            op.type = c[1]
                            op.device = c[2]
                            op.name = c[3]
                            op.implementation = c[4]
                            op.propagation = c[5]
                            op.iodata = c[6]
                            op.aux = c[7]
                            op.reserved = c[8]
                            op.benchdnn = c[9]
                            op.time = c[10]
                            self.ops.append(op)
        except IOError:
            print('{} not accessible'.format(self.args.logfile))

    def print_sequence(self, print_create=False):
        for op in self.ops:
            if not print_create and op.type.startswith('create'):
                continue
            print('{},{},{},{},{}'.format(op.type, op.name, op.device, op.benchdnn, op.time))

    def getOpList(self):
        return {op.name for op in self.ops}

    def analyze_exec(self):
        ops_time = {}
        ops_num = {}
        for op in self.getOpList():
            time = 0
            num = 0
            for op1 in list(filter(lambda x: x.name == op and x.type == 'exec', self.ops)):
                time += float(op1.time)
                num += 1
                # print('{}: {}, {}, {}, {}'.format(op1.name, op1.device, op1.iodata, op1.benchdnn, op1.time))
            ops_time[op] = time
            ops_num[op] = num
        ops_time = sorted(ops_time.items(), key=lambda item:item[1], reverse=True)
        # ops_time = collections.OrderedDict(ops_time)
        print('\nOperations Summary')
        for time in ops_time:
            print('{}: {:.2f}ms, {} time(s)'.format(time[0], time[1], ops_num[time[0]]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    args = parser.parse_args()
    dv = DNNL_Verbose(args)
    dv.load()
    print('version:     {}'.format(dv.version))
    print('cpu_runtime: {}'.format(dv.cpu_runtime))
    print('cpu_isa:     {}'.format(dv.cpu_isa))
    print('gpu_runtime: {}'.format(dv.gpu_runtime))
    print('gpu_engine:  {}'.format(dv.gpu_engine))
    dv.analyze_exec()
    # dv.print_sequence()

if __name__ == '__main__':
    main()
