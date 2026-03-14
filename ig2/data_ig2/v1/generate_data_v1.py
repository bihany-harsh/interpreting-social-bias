import json
import os

random_neg_adjs = json.load(open('../metadata/10_random_neg_adjs.json', 'r'))
neg_cer_adjs = json.load(open('../metadata/10_random_worse_adjs.json', 'r'))
neg_cest_adjs = json.load(open('../metadata/10_random_worst_adjs.json', 'r'))
random_pos_adjs = json.load(open('../metadata/10_random_pos_adjs.json', 'r'))
pos_cer_adjs = json.load(open('../metadata/10_random_better_adjs.json', 'r'))
pos_cest_adjs = json.load(open('../metadata/10_random_best_adjs.json', 'r'))
modifier_prompt_templates = json.load(open('modifier_prompt_template_v1.json', 'r'))
demographic_dict = json.load(open('../metadata/demographic_dict.json', 'r'))
modifier_dict = {
    'N': random_neg_adjs,
    'NCer': neg_cer_adjs,
    'NCest': neg_cest_adjs,
    'P': random_pos_adjs,
    'PCer': pos_cer_adjs,
    'PCest': pos_cest_adjs
}

for demographic_dimension in demographic_dict.keys():
    demo_data_dir = f"{demographic_dimension}_mini"
    for demo_id, demographic in enumerate(demographic_dict[demographic_dimension]):
        for modifier_type in modifier_dict.keys():
            sample_id = str(demographic_dimension)[:2].upper() + '-'
            sample_id += modifier_type
            sample_id += '-'
            sample_id += str(demo_id)
            # print('modifier_type', modifier_type)
            print('sample_id', sample_id)
            # os.makedirs(str(demographic_dimension), exist_ok=True)
            os.makedirs(str(demo_data_dir), exist_ok=True)
            # save_path = os.path.join(str(demographic_dimension),
            save_path = os.path.join(str(demo_data_dir),
                                     str(demographic) + '_' + str(modifier_type) + '_data.json')
            all_data = []
            for adj in modifier_dict[modifier_type]:
                bag_samples = []
                for prompt_template in modifier_prompt_templates:
                    sent_template = prompt_template[0]
                    generated_sent = sent_template.replace('[Demographic_Dimension]', demographic_dimension)
                    generated_sent = generated_sent.replace('[Modifier]', adj)
                    # print('generated_sent', generated_sent)
                    generated_sample = [
                        generated_sent,
                        f" {demographic}",
                        sample_id + '(' + demographic_dimension + '-' + demographic + ')'
                    ]
                    bag_samples.append(generated_sample)
                all_data.append(bag_samples)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4)
                