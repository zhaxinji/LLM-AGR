from config.configurator import configs
from trainer.utils import set_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from load_data.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner

if __name__ == '__main__':

    set_seed(2025)

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    logger = Logger()

    trainer = build_trainer(data_handler, logger)

    trainer.train(model)



