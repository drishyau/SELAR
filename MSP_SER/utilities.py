import os, yaml


class Logs:
    def __init__(self, exp_name):
        self.chkpt_folder = os.path.join(os.getcwd(), "RESULTS", exp_name.upper())
        if not os.path.isdir(self.chkpt_folder): os.makedirs(self.chkpt_folder)
        with open(os.path.join(self.chkpt_folder, "LOG_FILE.txt"), "w") as F:
            F.write(
                "{} Experiment Results\n".format(exp_name.upper())
            )
            F.write("\n")
        F.close()
    def write(self, *args, **kwargs):
        data = " ".join([str(i) for i in args])
        with open(os.path.join(self.chkpt_folder, "LOG_FILE.txt"), "a") as F:
            F.write(">>  "+data+"\n")
        F.close()


def GetYAMLConfigs(path):
    with open(path, 'r') as file:
        c = yaml.safe_load(file)
    return c