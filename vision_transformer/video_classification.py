import os
import glob
import tarfile
import imageio
import numpy as np
import pytorchvideo.data
import torch
import evaluate

from huggingface_hub import hf_hub_download
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


class DataLoader:
    def __init__(self):
        self.extract_data()
        self.define_lable_id_conversion()

    def extract_data(self):
        hf_dataset_identifier = "sayakpaul/ucf101-subset"
        filename = "UCF101_subset.tar.gz"
        file_path = hf_hub_download(
            repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
        )

        with tarfile.open(file_path) as t:
            t.extractall(".")

    def define_lable_id_conversion(self):
        all_video_file_paths = glob.glob(os.path.join("UCF101_subset", "*", "*"))
        class_labels = sorted(
            {str(path).split("/")[2] for path in all_video_file_paths}
        )
        self.label2id = {label: i for i, label in enumerate(class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        print(f"Unique classes: {list(self.label2id.keys())}.")


class DataVisualization:
    def __init__(self, dataset):
        self.dataset = dataset

    def unnormalize_img(self, img):
        img = (img * self.dataset.data_transform.std) + self.dataset.data_transform.mean
        img = (img * 255).astype("uint8")
        return img.clip(0, 255)

    def create_gif(self, video_tensor, filename="sample.gif"):
        frames = []
        for video_frame in video_tensor:
            frame_unnormalized = self.unnormalize_img(
                video_frame.permute(1, 2, 0).numpy()
            )
            frames.append(frame_unnormalized)
        kargs = {"duration": 0.25}
        imageio.mimsave(filename, frames, "GIF", **kargs)
        return filename

    def display_gif(self, video_tensor, gif_name="sample.gif"):
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        gif_filename = self.create_gif(video_tensor, gif_name)
        # return Image(filename=gif_filename)

    def visualize_video_sample(self):
        sample_video = next(iter(self.dataset.train_dataset))
        video_tensor = sample_video["video"]
        self.display_gif(video_tensor)


class Model:
    def __init__(self, data_loader):
        self.model_ckpt = "MCG-NJU/videomae-base"
        self.data_loader = data_loader
        self.define_image_processor()
        self.define_model()

    def define_image_processor(self):
        self.image_processor = VideoMAEImageProcessor.from_pretrained(self.model_ckpt)

    def define_model(self):
        self.mae_model = VideoMAEForVideoClassification.from_pretrained(
            self.model_ckpt,
            label2id=self.data_loader.label2id,
            id2label=self.data_loader.id2label,
            ignore_mismatched_sizes=True,
        )


class DataTransformer:
    def __init__(self, model):
        self.image_processor = model.image_processor
        self.model = model.mae_model
        self.define_parameters()
        self.define_tramsforms()

    def define_parameters(self):
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        self.resize_to = (height, width)

        self.num_frames_to_sample = self.model.config.num_frames
        self.sample_rate = 4
        self.fps = 30
        self.clip_duration = self.num_frames_to_sample * self.sample_rate / self.fps

    def define_tramsforms(self):
        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(self.resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        self.val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize(self.resize_to),
                        ]
                    ),
                ),
            ]
        )


class Dataset:
    def __init__(self, data_transform):
        self.data_transform = data_transform

        self.train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join("UCF101_subset", "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "random", self.data_transform.clip_duration
            ),
            decode_audio=False,
            transform=self.data_transform.train_transform,
        )

        self.val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join("UCF101_subset", "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.data_transform.clip_duration
            ),
            decode_audio=False,
            transform=self.data_transform.val_transform,
        )

        self.test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join("UCF101_subset", "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.data_transform.clip_duration
            ),
            decode_audio=False,
            transform=self.data_transform.val_transform,
        )

        print(
            self.train_dataset.num_videos,
            self.val_dataset.num_videos,
            self.test_dataset.num_videos,
        )


class ModelFineTuning:
    def __init__(self, model, dataset):
        self.model_name = model.model_ckpt.split("/")[-1]
        self.model = model.mae_model
        self.image_processor = model.image_processor
        self.train_dataset = dataset.train_dataset
        self.val_dataset = dataset.val_dataset
        self.define_training_arguments()
        self.define_trainer()

    def define_training_arguments(self):
        new_model_name = f"{self.model_name}-finetuned-ucf101-subset"
        num_epochs = 4
        batch_size = 8
        self.args = TrainingArguments(
            new_model_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=True,
            max_steps=(self.train_dataset.num_videos // batch_size) * num_epochs,
        )

    def define_trainer(self):
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["video"].permute(1, 0, 2, 3) for example in examples]
            )
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

    def fine_tune(self):
        return self.trainer.train()


def main():
    data_loader = DataLoader()
    model = Model(data_loader)
    data_transformer = DataTransformer(model)
    dataset = Dataset(data_transformer)

    data_visualizer = DataVisualization(dataset)
    data_visualizer.visualize_video_sample()

    model_fine_tuning = ModelFineTuning(model, dataset)


if __name__ == "__main__":
    main()
