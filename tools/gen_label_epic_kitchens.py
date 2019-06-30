import json
import os

import click
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset
import matplotlib.pyplot as plt


@click.command()
@click.option('--gulp-dir',
              type=str,
              default=None,
              help='EPIC-Kitchens gulp dir.')
@click.option('--interim-dir',
              type=str,
              default=None,
              help='/path/to/starter-kit-action-recognition/data/interim')
@click.option('--out', type=str, default=None, help='Output root folder.')
@click.option('--split-path',
              type=str,
              default=None,
              help='trainval.json path.')
def gen_label(gulp_dir, interim_dir, out, split_path):
    with open(split_path, 'r') as f:
        trainval = json.load(f)
    idxsplit = (len(trainval['train']) + len(trainval['val']))*[None]
    for i in trainval['train']:
        idxsplit[i] = 'train'
    for i in trainval['val']:
        idxsplit[i] = 'val'
    assert None not in idxsplit

    action_classes = {}
    class_counts = {}
    next_action_class = 0
    rgbviddata = EpicVideoDataset(f'{gulp_dir}/rgb_train', 'verb+noun')
    outputs = {'train': [], 'val': []}
    categories = []
    for i, seg in enumerate(rgbviddata.video_segments):
        parid = seg['participant_id']
        vidid = seg['video_id']
        nar = seg['narration'].replace(' ', '-')
        uid = seg['uid']
        reldir = f'{parid}/{vidid}/{vidid}_{uid}_{nar}'
        assert os.path.exists(f'{interim_dir}/{reldir}'), f'{interim_dir}/{reldir}'

        verb = seg['verb_class']
        noun = seg['noun_class']
        action = f'{verb},{noun}'
        if action in action_classes:
            classidx = action_classes[action]
            class_counts[action] += 1
        else:
            categories.append(f'{seg["verb"]} {seg["noun"]}')
            classidx = next_action_class
            action_classes[action] = classidx
            class_counts[action] = 1
            next_action_class += 1

        nframes = seg['num_frames']
        outputs[idxsplit[i]].append(f'{reldir} {nframes} {classidx}')

    assert len(set(categories)) == len(categories)

    with open(f'{out}/category.txt', 'w') as f:
        f.write('\n'.join(categories))

    with open(f'{out}/train_videofolder.txt', 'w') as f:
        f.write('\n'.join(outputs['train']))

    with open(f'{out}/val_videofolder.txt', 'w') as f:
        f.write('\n'.join(outputs['val']))

    class_counts = list(class_counts.values())
    class_counts.sort()
    plt.bar(range(0, len(class_counts)), class_counts)
    plt.savefig('action_class_histogram.png')


if __name__ == '__main__':
    gen_label()  # pylint:disable=no-value-for-parameter
