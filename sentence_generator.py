import random
from kb import load_kb


class SentenceGenerator:
    def __init__(self, kb):
        self.last_states_pred_dict = None
        self.last_selected_search_result = None
        self.kb = kb
        self.sent_type_mapping = {
            # TODO
        }

    def generate(self, states_pred_dict, sent_type, sent_groups, onto):
        # the index of onto is 1 greater than argmax
        sentence = ""
        # area food pricerange
        search_query = []

        if states_pred_dict.get("area") == self.last_states_pred_dict.get("area") and states_pred_dict.get(
                "food") == self.last_states_pred_dict.get("food") and states_pred_dict.get(
                "pricerange") == self.last_states_pred_dict.get("pricerange"):
            selected_search_result = self.last_selected_search_result
        else:
            for key, value in states_pred_dict:
                if not key.endswith("_req"):
                    search_query.append([key, onto.get(key)[value - 1]])
            search_result = list(self.kb.search_multi(search_query))
            if len(search_result) != 0:
                selected_search_result = search_result[0]
                self.last_selected_search_result = selected_search_result
                original_sent = random.choice(sent_groups[sent_type])
                original_words = original_sent.split(" ")
            elif len(search_result) == 0:
                original_sent = random.choice(sent_groups[str(int(41))])
                original_words = original_sent.split(" ")
        for original_word in original_words:
            if original_word == "<v.ADDRESS>":
                sentence = sentence + self.kb.get(selected_search_result).get('address') + " "
            elif original_word == "<v.AREA>":
                sentence = sentence + self.kb.get(selected_search_result).get('area') + " "
            elif original_word == "<v.FOOD>":
                sentence = sentence + self.kb.get(selected_search_result).get('food') + " "
            elif original_word == "<v.NAME>":
                sentence = sentence + self.kb.get(selected_search_result).get('name') + " "
            elif original_word == "<v.PHONE>":
                sentence = sentence + self.kb.get(selected_search_result).get('phone') + " "
            elif original_word == "<v.POSTCODE>":
                sentence = sentence + self.kb.get(selected_search_result).get('postcode') + " "
            elif original_word == "<v.PRICERANGE>":
                sentence = sentence + self.kb.get(selected_search_result).get('pricerange') + " "
            elif original_word == "<s.ADDRESS>":
                sentence = sentence + "address "
            elif original_word == "<s.AREA>":
                sentence = sentence + "area "
            elif original_word == "<s.FOOD>":
                sentence = sentence + "food "
            elif original_word == "<s.NAME>":
                sentence = sentence + "name "
            elif original_word == "<s.PHONE>":
                sentence = sentence + "phone "
            elif original_word == "<s.POSTCODE>":
                sentence = sentence + "postcode "
            elif original_word == "<s.PRICERANGE>":
                sentence = sentence + "pricerange "
            else:
                sentence = sentence + original_word + " "
        self.last_states_pred_dict = states_pred_dict
        return sentence
