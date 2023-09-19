__author__='mohamedelbadrashiny'

import aixplain.model_schemas.utils.metric_utils as utils
import re
import docs.user.samples.metric.diacritization_accuracy.src.utils as diacritization_utils

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import ClassificationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import ClassificationMetricInput
from aixplain.model_schemas.schemas.metric_output import ClassificationMetricOutput
from typing import Dict, List

class DiacritizationAccuracy(ClassificationMetric):
    def __clean(self, sentence:str):
        tmp = utils.remove_urls(sentence)
        tmp = utils.remove_emails(tmp)
        tmp = utils.remove_numbers(tmp)
        tmp = diacritization_utils.remove_non_arabic(tmp)
        tmp = utils.remove_emojis(tmp)
        tmp = diacritization_utils.remove_kashida_dagger_arabic(tmp)
        tmp = diacritization_utils.remove_superfluous_arabic(tmp)
        tmp = diacritization_utils.remove_default_diacritics_arabic(tmp)
        tmp = diacritization_utils.remove_punctuation(tmp)
        return tmp


    def __compare_diacritics(self, hyp, ref):
        w_err = 0
        w_err_without_case_ending = 0
        c_err = 0
        c_err_without_case_ending = 0
        case_ending_idx = len(ref) - 1
        for h, r in zip(hyp, ref):
            if h[1] != r[1]:  # i.e. the diacritics of this letter don't match
                c_err += 1
        if c_err > 0:
            w_err = 1
            if hyp[case_ending_idx][1] != ref[case_ending_idx][1]:  # i.e. the diacritics of the last letter don't match
                c_err_without_case_ending = c_err - 1
            else:
                c_err_without_case_ending = c_err
        if c_err_without_case_ending > 0:
            w_err_without_case_ending = 1

        return w_err, w_err_without_case_ending, c_err, c_err_without_case_ending


    def __count_errors(self, hyp, ref):
        hyp_words = hyp.split()
        ref_words = ref.split()
        w_err = 0
        w_err_without_case_ending = 0
        c_err = 0
        c_err_without_case_ending = 0
        w_num = len(ref_words)  # the number of reference words
        c_num = len(diacritization_utils.remove_diacritics(ref).replace(" ", ""))  # number of reference letters
        c_num_without_case_ending = c_num - w_num  # number of reference letters ignoring the case ending
        # we subtracted w_num because we have 1 case ending letter for each word

        if len(hyp_words) != len(ref_words):  # miss-aligned data. We will consider the whole sentence is wrong
            w_err = w_num
            w_err_without_case_ending = w_num
            c_err = c_num
            c_err_without_case_ending = c_num_without_case_ending
        else:  # i.e. correctly aligned words
            for w_h, w_r in zip(hyp_words, ref_words):
                hyp_letters_diacs = diacritization_utils.split_diacritics(w_h)
                ref_letters_diacs = diacritization_utils.split_diacritics(w_r)
                if len(hyp_letters_diacs) != len(
                        ref_letters_diacs):  # i.e. the hyp_words and the ref_words don't have the same number of
                    # letters. We will count this word as wrong
                    w_err += 1
                    w_err_without_case_ending += 1
                    c_err += len(ref_letters_diacs)
                    c_err_without_case_ending += len(ref_letters_diacs) - 1
                else:
                    w_e, w_e_no_cs_end, c_e, c_e_no_cs_end = self.__compare_diacritics(hyp_letters_diacs, ref_letters_diacs)
                    w_err += w_e
                    w_err_without_case_ending += w_e_no_cs_end
                    c_err += c_e
                    c_err_without_case_ending += c_e_no_cs_end

        return w_num, c_num, c_num_without_case_ending, w_err, w_err_without_case_ending, c_err, c_err_without_case_ending


    def __calculate_accuracy(self, w_ct, c_ct, c_ct_no_cs_end, w_err, w_err_no_cs_end, c_err, c_err_no_cs_end):
        accuracy = {'Word_accuracy': round(100 * (1 - w_err / w_ct), 2),
                    'Word_accuracy_ignoring_case_ending': round(100 * (1 - w_err_no_cs_end / w_ct), 2),
                    'Characters_accuracy': round(100 * (1 - c_err / c_ct), 2),
                    'Characters_accuracy_ignoring_case_ending': round(100 * (1 - c_err_no_cs_end / c_ct_no_cs_end), 2)}
        return accuracy
        

    def run_metric(self, request: Dict[str, List[ClassificationMetricInput]]) -> Dict[str, List[ClassificationMetricOutput]]:
        """Diacritization Accuracy function

        It takes:
        hypothesis: a list of diacritized sentences --> EX:
        ["فَرَاشَةً مُلَوِّنَةً تَطِيْرِ في البُسْتَانِ،", "تَحُطُّ فِي نُعُومَةً تَنْشُرَ السَّلاَمَ"]
        references: a list of the reference diacritization of the same sentences --> EX:
        ["فَرَاشَةٌ مُلَوَّنَةٌ تَطِيْرُ في البُسْتَانِ،", "تَحُطُّ فِي نُعُومَةٍ تَنْشُرُ السَّلاَمَ”]
        Returns the accuracy of:
        1- correct words and correct words while ignoring case-ending for each sentence and for the whole document
        2- correct characters and correct characters while ignoring case-ending for each sentence and for the whole document
        
        Assumptions and normalization: 
        1- the case-ending is always on the last letter of the word 
        2- this class ignores the default daicritics; Fatha followed by Alef, Kasra followed by Yaa, Dammah 
        followed by Waw, and Alef with Hamza Below followed by Kasra

        Args:
            request (Dict[str, List[ClassificationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[ClassificationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            hypotheses, references = inp.hypotheses, inp.references
            w_total_num = 0
            c_total_num = 0
            c_total_num_without_case_ending = 0
            w_total_err = 0
            w_total_err_without_case_ending = 0
            c_total_err = 0
            c_total_err_without_case_ending = 0

            sentences_results = []

            for hyp, ref in zip(hypotheses, references):
                hyp = self.__clean(hyp)
                ref = self.__clean(ref)
                w_ct = 1
                c_ct = 1
                c_ct_no_cs_end = 1
                w_err = 0
                w_err_no_cs_end = 0
                c_err = 0
                c_err_no_cs_end = 0
                w_ct, c_ct, c_ct_no_cs_end, w_err, w_err_no_cs_end, c_err, c_err_no_cs_end = self.__count_errors(hyp, ref)
                sentences_results.append(
                    self.__calculate_accuracy(w_ct, c_ct, c_ct_no_cs_end, w_err, w_err_no_cs_end, c_err, c_err_no_cs_end))
                w_total_num += w_ct
                c_total_num += c_ct
                c_total_num_without_case_ending += c_ct_no_cs_end
                w_total_err += w_err
                w_total_err_without_case_ending += w_err_no_cs_end
                c_total_err += c_err
                c_total_err_without_case_ending += c_err_no_cs_end
    
            total_accuracy = self.__calculate_accuracy(w_total_num, c_total_num, c_total_num_without_case_ending,
                                                    w_total_err, w_total_err_without_case_ending, c_total_err,
                                                    c_total_err_without_case_ending)

            return_dict = {'corpus-level': total_accuracy, 'sentence-level': sentences_results}
            output_dict = ClassificationMetricOutput(**{
                "data": return_dict['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = DiacritizationAccuracy(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])