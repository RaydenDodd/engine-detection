import pickle
import os
# Non-GUI setup
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
pickle_file_path = os.path.join(script_dir, '..', 'trained_models', 'label_to_category_all_mapping.pickle')
with open(pickle_file_path , 'rb') as file:
    brands_mapping = pickle.load(file)
continue_flag = False

brands_image_dict = {brand_name: f'photos/{brand_name.lower()}.png' for brand_name in brands_mapping.values()}


print(brands_mapping)


print("\nimage\n")

print(brands_image_dict)

results_accumulator = [0.02, 0.75, 0.03, 0.1, 0.55, 0.15, 0.33, 0.44, 0.12, 0.07, 0.65, 0.28, 0.82, 0.09, 0.5, 0.6, 0.2, 0.95, 0.4, 0.03]

sorted_results = sorted(range(len(results_accumulator)), key=lambda k: results_accumulator[k])

brand_1 = brands_mapping[sorted_results[-1]]
brand_2 = brands_mapping[sorted_results[-2]]
brand_3 = brands_mapping[sorted_results[-3]]

print(brand_1)
print(brand_2)
print(brand_3)