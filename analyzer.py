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
        self.dnn_version = ''
        self.dnn_cpu_runtime = ''
        self.dnn_cpu_isa = ''
        self.dnn_gpu_runtime = ''
        self.dnn_gpu_engine = ''
        self.dnn_ops = []

        self.mkl_version = ''
        self.mkl_ops = []

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
                                self.dnn_version = c[2]
                            if c[2].startswith('Detected ISA'):
                                self.dnn_cpu_isa = ','.join(c[2:]).replace('Detected ISA is ', '')
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
                            op.time = float(c[8])
                            self.dnn_ops.append(op)

                    if line.startswith('dnnl_verbose,'):
                        line = line.replace('\r', '')
                        line = line.replace('\n', '')
                        c = line.split(',')
                        if c[1] == 'info':
                            if len(c) == 3:
                                self.dnn_version = c[2]
                            else:
                                if c[2] == 'cpu':
                                    if c[3].startswith('runtime:'):
                                        self.dnn_cpu_runtime = c[3].split(':')[1]
                                    if c[3].startswith('isa:'):
                                        self.dnn_cpu_isa = c[3].split(':')[1]
                                if c[2] == 'gpu':
                                    if c[3].startswith('runtime:'):
                                        self.dnn_gpu_runtime = c[3].split(':')[1]
                                    if c[3] == 'engine':
                                        self.dnn_gpu_engine = c[5].split(':')[1]
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
                            op.time = float(c[10])
                            self.dnn_ops.append(op)

                    if line.startswith('MKL_VERBOSE'):
                        line = line.replace('\r', '')
                        line = line.replace('\n', '')
                        line = line[12:]
                        if line.startswith('oneMKL') or line.startswith('Intel(R) MKL'):
                            self.mkl_version = line
                        else:
                            c = line.split(' ')
                            alg_raw = c[0]
                            time_value = float(c[1][:-2])
                            time_unit = c[1][-2:]
                            if time_unit == 'ms':
                                time_value = time_value * 1000
                            op = Op()
                            op.name = alg_raw.split('(')[0]
                            op.time = time_value
                            self.mkl_ops.append(op)
        except IOError:
            print('{} not accessible'.format(self.args.logfile))

    def print_sequence(self, print_create=False):
        for op in self.dnn_ops:
            if not print_create and op.type.startswith('create'):
                continue
            print('{},{},{},{},{}'.format(op.type, op.name, op.device, op.benchdnn, op.time))

    def getOpList(self, lib='dnn'):
        if lib == 'dnn':
            return {op.name for op in self.dnn_ops}
        else:
            return {op.name for op in self.mkl_ops}

    def analyze_exec(self):
        if len(self.dnn_ops) > 0:
            ops_time = {}
            ops_num = {}
            for op in self.getOpList('dnn'):
                time = 0
                skip = 0
                oplist = list(filter(lambda x: x.name == op and x.type == 'exec', self.dnn_ops))
                num = len(oplist)
                if num % self.args.iters == 0:
                    t = num / self.args.iters
                    skip = t * (self.args.iters - 1)
                for i in range(num):
                    if i < skip:
                        continue
                    time += oplist[i].time
                    # print('{}: {}, {}, {}, {}'.format(op1.name, op1.device, op1.iodata, op1.benchdnn, op1.time))
                if num % self.args.iters == 0:
                    num = num / self.args.iters
                else:
                    num = num / float(self.args.iters)
                    time = time / self.args.iters
                ops_time[op] = time
                ops_num[op] = num
            ops_time = sorted(ops_time.items(), key=lambda item:item[1], reverse=True)
            # ops_time = collections.OrderedDict(ops_time)

            print('======================')
            print('onednn_version:     {}'.format(self.dnn_version))
            print('onednn_cpu_runtime: {}'.format(self.dnn_cpu_runtime))
            print('onednn_cpu_isa:     {}'.format(self.dnn_cpu_isa))
            print('onednn_gpu_runtime: {}'.format(self.dnn_gpu_runtime))
            print('onednn_gpu_engine:  {}'.format(self.dnn_gpu_engine))
            print('')
            print('op name;dur (ms);num occurrence')
            for time in ops_time:
                print('{};{:.2f};{:.2f}'.format(time[0], time[1], ops_num[time[0]]))
            print('')

        if len(self.mkl_ops) > 0:
            ops_time = {}
            ops_num = {}
            for op in self.getOpList('mkl'):
                time = 0
                skip = 0
                oplist = list(filter(lambda x: x.name == op, self.mkl_ops))
                num = len(oplist)
                if num % self.args.iters == 0:
                    t = num / self.args.iters
                    skip = t * (self.args.iters - 1)
                for i in range(num):
                    if i < skip:
                        continue
                    time += oplist[i].time
                    # print('{}: {}, {}, {}, {}'.format(op1.name, op1.device, op1.iodata, op1.benchdnn, op1.time))
                if num % self.args.iters == 0:
                    num = num / self.args.iters
                else:
                    num = num / float(self.args.iters)
                    time = time / self.args.iters
                ops_time[op] = time
                ops_num[op] = num
            ops_time = sorted(ops_time.items(), key=lambda item:item[1], reverse=True)
            # ops_time = collections.OrderedDict(ops_time)

            print('======================')
            print('onemkl_version:     {}'.format(self.mkl_version))
            print('')
            print('op name;dur (ms);num occurrence')
            for time in ops_time:
                print('{};{:.2f};{:.2f}'.format(time[0], time[1]/1000, ops_num[time[0]]))
            print('')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    parser.add_argument("--iters", default=1, type=int)
    args = parser.parse_args()
    dv = DNNL_Verbose(args)
    dv.load()
    dv.analyze_exec()
    # dv.print_sequence()

if __name__ == '__main__':
    main()
