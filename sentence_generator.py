import random


class SentenceGenerator:
    def __init__(self, kb, onto, sent_groups):
        self.last_states_pred_dict = None
        self.last_selected_search_result = None
        self.kb = kb
        self.onto = onto
        self.sent_groups = sent_groups
        self.sent_type_mapping = {
            # TODO
        }

    def generate(self, states_pred_dict, sent_type):
        # the index of onto is 1 greater than argmax
        sentence = ""
        # possible search fields: area, food, and pricerange
        search_query = []
        selected_search_result = ""
        original_sent = random.choice(self.sent_groups[sent_type])
        original_words = original_sent.split(" ")

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
            if len(search_result) != 0:
                selected_search_result = search_result[0]
                self.last_states_pred_dict = states_pred_dict
                self.last_selected_search_result = selected_search_result
            elif len(search_result) == 0:
                self.last_selected_search_result = None
                self.last_selected_search_result = None
                original_sent = random.choice(self.sent_groups[str(int(41))])
                original_words = original_sent.split(" ")
            print search_query
            print search_result
            print self.kb.get(selected_search_result)
        for original_word in original_words:
            if original_word.startswith("<v.ADDRESS>"):
                sentence = sentence + self.kb.get(selected_search_result).get('address') + " "
            elif original_word.startswith("<v.AREA>"):
                sentence = sentence + self.kb.get(selected_search_result).get('area') + " "
            elif original_word.startswith("<v.FOOD>"):
                sentence = sentence + self.kb.get(selected_search_result).get('food') + " "
            elif original_word.startswith("<v.NAME>"):
                sentence = sentence + self.kb.get(selected_search_result).get('name') + " "
            elif original_word.startswith("<v.PHONE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('phone') + " "
            elif original_word.startswith("<v.POSTCODE>"):
                sentence = sentence + self.kb.get(selected_search_result).get('postcode') + " "
            elif original_word.startswith("<v.PRICERANGE>"):
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
            else:
                sentence = sentence + original_word + " "
        return sentence
