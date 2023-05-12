from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class Logger:
    def __init__(self, distributed: bool, logdir: str):
        self.distributed = distributed
        self.tb = SummaryWriter(log_dir=logdir)
        self.metrics = {}

    def log(self, msg, **kwargs):
        if not self.distributed.distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    def updateMetric(self, key, value, pb):
        if key not in self.metrics.keys():
            count = 0
            total = 0
            self.metrics[key] = (total, count)
        total, count = self.metrics[key]
        total += value
        count += 1
        self.metrics[key] = (total, count)
        if key == "loss":
            pb.set_postfix({key: self.avg(key)})

    def avg(self, key):
        total, count = self.metrics[key]
        return total / count

    def endMetricEpoch(self, epoch):
        for key in self.metrics.keys():
            self.tb.add_scalar(key, self.avg(key), epoch)
        self.tb.flush()
        self.metrics = {}

    def tbStat(self, tag, value, epoch):
        self.tb.add_scalar(tag, value, epoch)
        self.tb.flush()

    def tbImage(self, tag, images, epoch):
        self.tb.add_images(tag, images, epoch)
        self.tb.flush()
