import os

from tensorflow import keras

from models.hourglass import heatmap_mean_accuracy, load_model, parse_command_line
from utils.heatmap_dataset import HeatmapBoxCarsDataset
from utils.gpu import set_gpus


def train():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_gpus()

    scales = [0.03, 0.1, 0.3, 1.0]

    model, snapshot_dir_name, snapshot_dir_path = load_model(args, scales)

    adam = keras.optimizers.Adam(args.lr)
    model.compile(adam, 'MSE', metrics=[heatmap_mean_accuracy(args.batch_size, args.heatmap_size, len(scales) * 2)])

    print(model.summary())

    print("Loading dataset!")
    train_dataset = HeatmapBoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    print("Loaded training dataset with {} samples".format(len(train_dataset.instance_list)))
    val_dataset = HeatmapBoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    print("Loaded val dataset with {} samples".format(len(val_dataset.instance_list)))


    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshot_dir_path, 'model.{epoch:03d}.h5')),
                 keras.callbacks.TensorBoard(log_dir=os.path.join('logs', snapshot_dir_name))]

    print("Workers: ", args.workers)
    print("Use multiprocessing: ", args.workers > 1)
    print("Starting training with lr: {}".format(args.lr))


    model.fit_generator(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=callbacks, initial_epoch=args.resume, workers=args.workers, use_multiprocessing=args.workers > 1)

    if args.shutdown:
        os.system('sudo poweroff')


if __name__ == '__main__':
    train()