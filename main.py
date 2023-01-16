import os
import argparse
import getpass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image


from util import util, load_data
from model import data, discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Process Data')
    # Basic Setting.
    parser.add_argument('--data_path', type=str, required=False, default='Projects/Data/', help='Data dir')
    parser.add_argument('--pp_ap_data_path', type=str, required=False, default='PM_2 SPECS DAT files',
                        help='Pressure and Anteroposterior shear stress data dir')
    parser.add_argument('--tem_path', type=str, required=False, default='Landscape_Aligned_Dataset',
                        help='Temperature data dir')
    # parser.add_argument('--ap_label_path', type=str, required=False, default='PeakLocation.csv',
    #                     help='Anteroposterior shear peak ')
    parser.add_argument('--folder_name', type=str, required=False, default='./PTA-GAN/', help='Output folder dir')
    parser.add_argument('--gpu', type=str, default='1', help='Cuda visible device.')

    # Hyper Parameters.
    # parser.add_argument('--STEP', type=str, default=37, help='Step number')
    parser.add_argument('--EPOCH', type=int, default=100, help='Epoch numbers')
    parser.add_argument('--BATCH_SIZE', type=int, default=10, help='Batch size')
    parser.add_argument('--lamda', type=int, default=100, help='Masked loss weights')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Training rate')
    parser.add_argument('--model', type=str, default='PTA-GAN.pth', help='Output model name')
    parser.add_argument('--generator', type=str, default='PTA-GAN',
                        help='Generator: P-GAN, T-GAN, PT-GAN, PTL-GAN, TPA-GAN, PTA-GAN')


    opt = parser.parse_args()
    print(opt)
    return opt

def main():
    args = parse_args()
    username = getpass.getuser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """ get pressure&shear and temperature file paths """
    data_path = args.data_path
    pp_ap_data_path = args.pp_ap_data_path
    tem_path = args.tem_path
    # ap_label_path = 'PeakLocation.csv'
    data_path = os.path.join('/home/' + username, data_path)
    pp_ap_data_path = os.path.join(data_path, pp_ap_data_path)
    tem_path = os.path.join(data_path, tem_path)
    # ap_label_path = os.path.join(data_path, ap_label_path)
    print(data_path)
    print(pp_ap_data_path)
    print(tem_path)

    """ load pressure & shear and temperature data """
    # pp and ap file name style: 01_VDN (02_VDN Left 2 ap image has problem)
    pp_ap_label, pp_dataset, ap_dataset, mask_dataset = load_data.readPP_AP(pp_ap_data_path)
    train_tem, tem_label = load_data.readDeltaTem(tem_path, pp_dataset, pp_ap_label)

    pp_dataset = np.array(pp_dataset)
    ap_dataset = np.array(ap_dataset)
    train_tem = np.array(train_tem)
    mask_dataset = np.array(mask_dataset)

    """ Set hyper parameters """
    EPOCH = args.EPOCH  # train the training data n times, to save time, we just train 1 epoch
    BATCH_SIZE = args.BATCH_SIZE
    learning_rate = args.lr  # learning rate
    # N_TEST_IMG = 2
    lamda = args.lamda
    train_ratio = args.train_rate
    gan_mode = 'lsgan'

    folder_name = args.folder_name
    model = args.model
    generator = args.generator
    train_folder = folder_name + 'train/'
    test_folder = folder_name + 'test/'
    model_name = folder_name + model

    """ build custom train and test dataset """
    dataset = data.AP_TEM_Dataset(pp_dataset, ap_dataset, train_tem, mask_dataset)

    # randomly split
    n = len(dataset)
    train_size = int(train_ratio * n)
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(trainset, batch_size=5, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    STEP = len(test_loader)
    # print(STEP)

    """ initialize GAN model """
    # define discriminator
    netD = discriminator.PixelDiscriminator(input_nc=2).cuda()
    # define generator
    if generator == 'TPA-GAN':
        model = generator.AttentionGenerator(input_nc=1, output_nc=1, base='T').cuda()
    elif generator == 'P-GAN':
        model = generator.Generator(input_nc=1, output_nc=1).cuda()
    elif generator == 'T-GAN':
        model = generator.AGenerator(input_nc=1, output_nc=1, base='T').cuda()
    elif generator == 'PT-GAN':
        model = generator.PTGenerator(input_nc=1, output_nc=1).cuda()
    elif generator == 'PTL-GAN':
        model = generator.PTLGenerator(input_nc=1, output_nc=1).cuda()
    else:
        # default generator: 'PTA-GAN'
        model = generator.AttentionGenerator(input_nc=1, output_nc=1).cuda()

    # define loss functions
    criterionGAN = discriminator.GANLoss(gan_mode).cuda()
    criterionL1 = torch.nn.L1Loss()
    criterion = nn.MSELoss()
    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
    optimizer_G = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    """ train model """
    for epoch in range(EPOCH):
        for step, (pp, ap, tem, mask) in enumerate(train_loader):
            pp = pp.cuda()
            ap = ap.cuda()
            tem = tem.cuda()
            mask = mask.cuda()
            if generator == 'P-GAN':
                decoder = model(pp)
            elif generator == 'T-GAN':
                decoder = model(tem)
            else:
                # default
                decoder = model(pp, tem)
            # decoder = model(pp, tem)

            L1, L2 = util.L1L2(decoder, mask)
            L1_ap, L2_ap = util.L1L2(ap, mask)

            """ update D """
            util.set_requires_grad(netD, True)  # enable backprop for D
            optimizer_D.zero_grad()  # set D's gradients to zero

            """Calculate GAN loss for the discriminator"""
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((ap, decoder),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = (fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((ap, ap), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()

            optimizer_D.step()  # update D's weights

            """update G"""
            util.set_requires_grad(netD, False)  # D requires no gradients when optimizing G
            optimizer_G.zero_grad()  # set G's gradients to zero
            """Calculate GAN and L1 L2 loss for the generator"""
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((ap, decoder), 1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(L1, L1_ap)  # foot area L1 error
            loss_G_L2 = criterion(L2, L2_ap)  # mean square error
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + lamda * (loss_G_L1 + loss_G_L2)
            loss_G.backward()

            optimizer_G.step()  # udpate G's weights

        if epoch % 2 == 0:
            # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            print('epoch [{}/{}], loss:{:.6f}, l1_loss:{:.6f}, l2_loss:{:.6f}, loss_D:{:.6f}'
                  .format(epoch + 1, EPOCH, loss_G.cpu().data.numpy(), loss_G_L1.cpu().data.numpy(),
                          loss_G_L2.cpu().data.numpy(), loss_D.cpu().data.numpy()))
            pic = util.to_img(decoder.cpu().data)
            pic2 = util.to_img(ap.cpu().data)
            save_image(pic, train_folder + 'decoder/image_{}.png'.format(epoch))
            save_image(pic2, train_folder + 'original/image_{}.png'.format(epoch))
            torch.save(model.state_dict(), model_name)
    # torch.save(model.state_dict(), './non_local_autoencoder_v2.pth')

    """ test model """
    for step, (pp, ap, tem, mask) in enumerate(test_loader):
        pp = pp.cuda()
        ap = ap.cuda()
        tem = tem.cuda()
        mask = mask.cuda()
        if generator == 'P-GAN':
            decoder = model(pp)
        elif generator == 'T-GAN':
            decoder = model(tem)
        else:
            # default
            decoder = model(pp, tem)
        # decoder = model(pp, tem)

        L1, L2 = util.L1L2(decoder, mask)
        L1_ap, L2_ap = util.L1L2(ap, mask)

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((ap, decoder),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = (fake_AB.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((ap, ap), 1)
        pred_real = netD(real_AB)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        fake_AB = torch.cat((ap, decoder), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(L1, L1_ap)  # foot area L1 error
        loss_G_L2 = criterion(L2, L2_ap)  # mean square error
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + lamda * (loss_G_L1 + loss_G_L2)

        # l2_loss = criterion(L2, L2_ap)  # mean square error
        # l1_loss = criterionL1(L1, L1_ap)  # foot area L1 error
        #
        # loss = lamda * l1_loss + l2_loss
        print('step [{}/{}], loss:{:.6f}, l1_loss:{:.6f}, l2_loss:{:.6f}, loss_D:{:.6f}'
              .format(step + 1, STEP, loss_G.cpu().data.numpy(), loss_G_L1.cpu().data.numpy(),
                      loss_G_L2.cpu().data.numpy(), loss_D.cpu().data.numpy()))

        util.save_csv(pp, test_folder + 'pp/test_{}.csv'.format(step), MIN=0.0, MAX=1100000.0)
        util.save_csv(ap, test_folder + 'original/test_{}.csv'.format(step))
        util.save_csv(tem, test_folder + 'tem/test_{}.csv'.format(step), MIN=5.0, MAX=45.0)
        util.save_csv(decoder, test_folder + 'decoder/test_{}.csv'.format(step))


if __name__ == '__main__':
    main()