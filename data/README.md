# Place real images here, either layout works:
#   data/TrashNet/cardboard/*.jpg ... (class-wise folders), or
#   flat files like img_plastic_12.jpg
# Classes: cardboard, glass, metal, paper, plastic, trash.
# Without data the pipeline uses a synthetic fallback (same 6 classes).
#
# Sources merged automatically (PPT: data from multiple sources):
#   data/TrashNet/ - TrashNet resized (public) + our own metal photos (157)
#   data/Manual/   - extra collected photos: 100 each of
#                    cardboard, glass, paper, plastic, trash
#                    (public imagefolder set, HuggingFace indu22/Waste_classification)
