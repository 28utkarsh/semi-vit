import os
import argparse
import shutil
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset setup script")
    parser.add_argument("--data_dir", "-dd", type=str, required=True,
                        help="Path to the data directory")
    parser.add_argument("--output_dir", "-od", type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--val_perc", "-vp", type=float, default=0.1,
                        help="Percentage of validation data")
    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        raise RuntimeError("Data directory is missing!")
    
    os.makedirs(args.output_dir)

    return args

def main(args):
    labeled_file = os.path.join(args.data_dir, "train_labeled.csv")
    categories = os.path.join(args.data_dir, "categories.csv")
    images_dir = os.path.join(args.data_dir, 'train', 'labeled')
    dest_train_dir = os.path.join(args.output_dir, "train")
    dest_val_dir = os.path.join(args.output_dir, "val")
    train_index_file = os.path.join(args.output_dir, "indexes", "train_index_file.csv")
    ulabeled_index_file = os.path.join(args.output_dir, "indexes", "train_unlabeled_index_file.csv")
    val_index_file = os.path.join(args.output_dir, "indexes", "val_index_file.csv")

    ulabeled_src = os.path.join(args.data_dir, 'train', 'unlabeled')

    class_mapping_dict = {}

    with open(categories, 'r') as reader:
        for line_idx, line in enumerate(reader):
            if line_idx==0:
                continue
            class_idx, class_name = line.strip().split(",")
            os.makedirs(os.path.join(dest_train_dir, class_idx))
            os.makedirs(os.path.join(dest_val_dir, class_idx))
            class_mapping_dict[class_idx] = class_name

    all_files = {}
    with open(labeled_file, 'r') as reader:
        for line_idx, line in enumerate(reader):
            if line_idx==0:
                continue
            image_name, class_idx = line.strip().split(",")
            if class_idx in all_files:
                all_files[class_idx].append(image_name)
            else:
                all_files[class_idx] = [image_name]

    train_files = []
    val_files = []
    for class_idx in all_files.keys():
        num_val_files = int(len(all_files[class_idx]) * args.val_perc)
        values = all_files[class_idx]
        for _ in range(10):
            random.shuffle(values)
        val_files_per_class = values[:num_val_files]
        train_files_per_class = values[num_val_files:]
        for image_name in train_files_per_class:
            train_files.append((class_idx, os.path.join(class_idx, image_name)))
            src = os.path.join(images_dir, image_name)
            dest = os.path.join(dest_train_dir, class_idx, image_name)
            shutil.copy(src, dest)
        
        for image_name in val_files_per_class:
            val_files.append((class_idx, os.path.join(class_idx, image_name)))
            src = os.path.join(images_dir, image_name)
            dest = os.path.join(dest_val_dir, class_idx, image_name)
            shutil.copy(src, dest)
    
    with open(os.path.join(args.output_dir, "mapping_dict.pkl"), 'wb') as writer:
        pickle.dump(class_mapping_dict, writer, protocol=pickle.HIGHEST_PROTOCOL)

    
    os.makedirs(os.path.join(args.output_dir, "indexes"))
    train_class_idx = 0
    with open(train_index_file, 'w') as writer:
        writer.write("Index,ImageID\n")
        for _, file_path in train_files:
            writer.write("{},{}\n".format(train_class_idx, file_path))
            train_class_idx += 1
    
    # val_class_idx = 0
    # with open(val_index_file, 'w') as writer:
    #     writer.write("Index,ImageID\n")
    #     for _, file_path in val_files:
    #         writer.write("{},{}\n".format(val_class_idx, file_path))
    #         val_class_idx += 1

    all_unlabled_images = os.listdir(ulabeled_src)
    with open(ulabeled_index_file, 'w') as writer:
        writer.write("Index,ImageID\n")
        for filename in all_unlabled_images:
            dest_ulabeled = os.path.join(dest_train_dir, '0', filename)
            src_ulabeled = os.path.join(ulabeled_src, filename)
            shutil.copy(src_ulabeled, dest_ulabeled)
            writer.write("{},{}\n".format(train_class_idx, os.path.join('0', filename)))
            train_class_idx += 1

    
    # Verify which class doesn't have any data in it
    for folder_name in os.listdir(dest_train_dir):
        if len(os.listdir(os.path.join(dest_train_dir, folder_name))) == 0:
            print("Training {} is empty.".format(folder_name))
            shutil.rmtree(os.path.join(dest_train_dir, folder_name))
    
    for folder_name in os.listdir(dest_val_dir):
        if len(os.listdir(os.path.join(dest_val_dir, folder_name))) == 0:
            print("Validation {} is empty.".format(folder_name))
            shutil.rmtree(os.path.join(dest_val_dir, folder_name))
    print("Done")

if __name__ == "__main__":

    main(parse_args())
