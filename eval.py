from utils import *
#from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import numpy as np
from pdb import set_trace
from matplotlib import pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
#data_folder = './'
#keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
#batch_size = 64
#workers = 4
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
#checkpoint = torch.load(checkpoint)
#model = checkpoint['model']
#model = model.to(device)

# Switch to eval mode
#model.eval()

# Load test data
#test_dataset = PascalVOCDataset(data_folder,
#                                split='test',
#                                keep_difficult=keep_difficult)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    # true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    precisions_dict = {}
    with torch.no_grad():
        # Batches
        #for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            
            #set_trace()
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            #difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            #true_difficulties.extend(difficulties)

        # Calculate mAP
        # APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        mAPs = {}
        for threshold in np.arange(0.5, 0.95, 0.05):  
            precisions, APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, threshold)
            threshold = "%.2f" %threshold
            
            mAPs[threshold] = mAP 
            precisions_dict[threshold] = precisions

    # Print AP for each class
    pp.pprint(APs)
    print("\nMean Average Precision (mAP@.5): %.3f" % mAPs["0.50"])
    #set_trace()
    print("\nMean Average Precision (mAP@.7): %.3f" % mAPs["0.70"])
    #set_trace()
    print("\nMean Average Precision (mAP@.9): %.3f" % mAPs["0.90"])
    mean_mAPs = sum(mAPs.values())/len(mAPs)
    print("\nMean Average Precision (mAP@[.5:.95]): %.3f" % mean_mAPs)
    #set_trace()
    
    fig_0_50 = plt.figure
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("x here")
    plt.plot(x, precisions_dict["0.50"], label="threshold 0.5")
    print("plot here")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    print("finish labeling")
    fig_0_50.savefig("fig_0_50.png")
    print("saved figure") 
    # set_trace()
    

if __name__ == '__main__':
    evaluate(test_loader, model)
