import os
import torch


def read_files(feat_dir):
  file_list = []
  for root, dirs, files in os.walk(feat_dir, followlinks=True):
    for name in files:
      if name.endswith('.pt'):
        file_list.append(name)
  return file_list


def check_shape(file_list):
  # crossatt  decoder_dist  encoderout  text_dist  textemb
  for file in file_list:
    crossatt_feat = torch.load(os.path.join('data/MSDM/crossatt/', file))
    decoder_dist_feat = torch.load(os.path.join('data/MSDM/decoder_dist/', file))
    encoderout_feat = torch.load(os.path.join('data/MSDM/encoderout/', file))
    text_dist_feat = torch.load(os.path.join('data/MSDM/text_dist/', file))
    textemb_feat = torch.load(os.path.join('data/MSDM/textemb/', file))
    print("crossatt:{} | decoder_dist:{} | text_dist:{} | textemb:{} | encoderout:{}".format(crossatt_feat.shape, decoder_dist_feat.shape, text_dist_feat.shape, textemb_feat.shape, encoderout_feat.shape))
    length = crossatt_feat.shape[0]
    if decoder_dist_feat.shape[1] != length or text_dist_feat.shape[0] != length or textemb_feat.shape[1] != length:
      print('shape not match:', file)
    print('---------------------------------------')


if __name__ == '__main__':
  feat_dir = 'data/MSDM/crossatt/'
  file_list = read_files(feat_dir)
  check_shape(file_list)