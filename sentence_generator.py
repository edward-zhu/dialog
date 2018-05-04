import random


class SentenceGenerator:
    def __init__(self, kb, onto, sent_groups):
        self.last_states_pred_dict = None
        self.last_selected_search_result = None
        self.kb = kb
        self.onto = onto
        self.sent_groups = sent_groups
        self.sent_type_mapping = {
            "132":"137",
            "136":"137",
            "138":"137",
            "24":"41",
            "20":"137",
            "161":"41",
            # "27":"137",
            "21":"41",
            "8":"137",
            "96":"41",
            "120":"41",
            "122":"137",
            "123":"137",
            "124":"137",
            "126":"41",
            "194":"137",
            "197":"137",
            "196":"137",
            "191":"137",
            "193":"137",
            "115":"137",
            "117":"137",
            "116":"137",
            "111":"41",
            "110":"41", # An interesting group ...
            "176":"137", # Another interesting group
            "82":"137",
            "86":"137",
            "118":"137",
            "178":"137",
            "108":"137",
            "109":"137",
            "103":"137",
            "100":"137",
            "30":"137",
            "37":"41",
            "35":"137",
            # "34":"137",
            "60":"137",
            "65":"137",
            "68":"137",
            "175":"137",
            "173":"137",
            "171":"137",
            "170":"137",
            "182":"137",
            "183":"137",
            "180":"137",
            "181":"137",
            "6":"137",
            "99":"137",
            "163":"137",
            "15":"137",
            # "14":"137",
            "17":"137",
            "152":"137",
            "158":"41",
            "78":"41",
            "148":"137",
            "144":"137",
            "47":"192",
            "112":"150"
        }

    def generate(self, states_pred_dict, sent_type):
        # the index of onto is 1 greater than argmax
        sentence = ""
        # possible search fields: area, food, and pricerange
        search_query = []
        search_result = []
        selected_search_result = ""
        print sent_type
        if sent_type in self.sent_type_mapping.keys():
            sent_type = self.sent_type_mapping[sent_type]
        print sent_type
        original_sent = random.choice(self.sent_groups[sent_type])
        original_words = original_sent.split(" ")

        for key, value in states_pred_dict.items():
            if key == "food":
                record_food_type = self.onto.get(key)[value[0] - 1]
            if key == "area":
                record_area_type = self.onto.get(key)[value[0] - 1]
            if key == "pricerange":
                record_pricerange_type = self.onto.get(key)[value[0] - 1]

        if self.last_states_pred_dict is not None \
                and self.last_states_pred_dict is not None \
                and states_pred_dict.get("area")[0] == self.last_states_pred_dict.get("area")[0] \
                and states_pred_dict.get("food")[0] == self.last_states_pred_dict.get("food")[0] \
                and states_pred_dict.get("pricerange")[0] == self.last_states_pred_dict.get("pricerange")[0]:
            selected_search_result = self.last_selected_search_result
        else:
            for key, value in states_pred_dict.items():
                if not key.endswith("_req") and value[0] != 0:
                    search_query.append([key, self.onto.get(key)[value[0] - 1]])
            search_result = list(self.kb.search_multi(search_query))
            print search_query
            print search_result
            if len(search_result) != 0:
                search_result_length = len(search_result)
                selected_search_result = search_result[random.randint(0,search_result_length - 1)]
                self.last_states_pred_dict = states_pred_dict
                self.last_selected_search_result = selected_search_result
                print self.kb.get(selected_search_result)
                # has results but the sent_type is 119
                # mapping 119 -> 137
                if sent_type == "119":
                    sent_type = "137"
                    print sent_type
                    original_sent = random.choice(self.sent_groups[sent_type])
                    original_words = original_sent.split(" ")
            elif len(search_result) == 0:
                self.last_states_pred_dict = None
                self.last_selected_search_result = None
                original_sent = random.choice(self.sent_groups[str(int(41))])
                original_words = original_sent.split(" ")
        for original_word in original_words:
            if original_word.startswith("<v.ADDRESS>"):
                sentence = sentence + self.kb.get(selected_search_result).get('address') + " "
            elif original_word.startswith("<v.AREA>"):
                if len(search_result) == 0 and self.last_states_pred_dict is None:
                    sentence = sentence + record_area_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('area') + " "
            elif original_word.startswith("<v.FOOD>"):
                if len(search_result) == 0 and self.last_states_pred_dict is None:
                    sentence = sentence + record_food_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('food') + " "
            elif original_word.startswith("<v.NAME>"):
                sentence = sentence + self.kb.get(selected_search_result).get('name') + " "
            elif original_word.startswith("<v.PHONE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('phone') + " "
            elif original_word.startswith("<v.POSTCODE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('postcode') + " "
            elif original_word.startswith("<v.PRICERANGE>"):
                if len(search_result) == 0:
                    sentence = sentence + record_pricerange_type + " "
                else:
                    sentence = sentence + self.kb.get(selected_search_result).get('pricerange') + " "
            elif original_word.startswith("<s.ADDRESS>"):
                sentence = sentence + "address "
            elif original_word.startswith("<s.AREA>"):
                sentence = sentence + "area "
            elif original_word.startswith("<s.FOOD>"):
                sentence = sentence + "food "
            elif original_word.startswith("<s.NAME>"):
                sentence = sentence + "name "
            elif original_word.startswith("<s.PHONE>"):
                sentence = sentence + "phone "
            elif original_word.startswith("<s.POSTCODE>"):
                sentence = sentence + "postcode "
            elif original_word.startswith("<s.PRICERANGE>"):
                sentence = sentence + "pricerange "
            elif original_word == "ly":
                sentence = sentence.strip() + "ly "
            else:
                sentence = sentence + original_word + " "
        return sentence
