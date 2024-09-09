import os

# tc command
try:
    IF = os.environ["CONCIERGE_IF"]
except KeyError:
    # if running locally, use lo
    # IF= "lo" # this would screw you up
    # IF = "eno1"
    IF = "eno1"

IFB = "ifb0"

TC = ["sudo", "tc"]

TC_QDISC = [*TC, "qdisc"]
TC_QDISC_DEL = [*TC_QDISC, "del", "dev"]
TC_QDISC_ADD = [*TC_QDISC, "add", "dev"]
TC_QDISC_REPLACE = [*TC_QDISC, "change", "dev"]

TC_FILTER = [*TC, "filter"]
TC_FILTER_DEL = [*TC_FILTER, "del", "dev"]
TC_FILTER_ADD = [*TC_FILTER, "add", "dev"]
TC_FILTER_REPLACE = [*TC_FILTER, "change", "dev"]

TC_CLASS = [*TC, "class"]
TC_CLASS_DEL = [*TC_CLASS, "del", "dev"]
TC_CLASS_ADD = [*TC_CLASS, "add", "dev"]
TC_CLASS_REPLACE = [*TC_CLASS, "change", "dev"]

INSERT_KMODULE = ["sudo", "modprobe", "ifb", "numifbs=1"]
IFB_UP = ["sudo", "ip", "link", "set", "dev", "ifb0", "up"]

CPU_SHR = "cpu.shares"

CG_CLR = ["sudo", "cgclear"]
CG_DEL = ["sudo", "cgdelete"]
CG_CREATE = ["sudo", "cgcreate", "-g"]
CG_SET = ["sudo", "cgset", "-r"]
CG_MOV = ["sudo", "cgclassify", "-g"]
