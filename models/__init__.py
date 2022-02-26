def create_forward(opt, *argv):
    print(opt.model)
    from .exp_trainer import ExperimentNet
    model = ExperimentNet()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

