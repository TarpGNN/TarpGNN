from volleyball import *

import pickle


def return_dataset(cfg):
    if cfg.dataset_name == 'volleyball':
        train_anns = volley_read_dataset(cfg, cfg.train_seqs)
        train_frames = volley_all_frames(train_anns)

        test_anns = volley_read_dataset(cfg, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        all_tracks = load_tracks(all_anns)
        training_set = VolleyballDataset(all_anns, all_tracks, train_frames,
                                         cfg.data_path, cfg.image_size, cfg.out_size, num_before=cfg.num_before,
                                         num_after=cfg.num_after, is_training=True,
                                         is_finetune=(cfg.training_stage == 1))

        validation_set = VolleyballDataset(all_anns, all_tracks, test_frames,
                                           cfg.data_path, cfg.image_size, cfg.out_size, num_before=cfg.num_before,
                                           num_after=cfg.num_after, is_training=False,
                                           is_finetune=(cfg.training_stage == 1))

    else:
        assert False

    print('Reading dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))

    return training_set, validation_set
