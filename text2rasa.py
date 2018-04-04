# -*- coding: utf-8 -*-
import spacy
import json

class Text2Rasa:
  """Perform various heuristics on an input text using Spacy's dependency tree
   to create one or more training files for use with Rasa NLU.
  """

  def __init__(self, text, json_file):
    self.text = unicode(text)
    self.json_file = json_file
    self.parsed_json = dict()
    
  def do_spacy_parse(self):
    """do_spacy_parse and return a spacy doc object"""
    nlp = spacy.load('en')
    self.spacy_doc = nlp(self.text)

  def read_training_file(self):
    try:
      with open(self.json_file) as training_data_file:
        training_data = training_data_file.read()
        self.parsed_json = json.loads(training_data) 
    except IOError:
      print "File " + self.json_file + "not found, starting with no training data."
      # no existing file so set up the correct data structure
      self.parsed_json = dict(rasa_nlu_data=dict(common_examples=list()))
    
  def write_training_file(self):
    #TODO error handling
    try:
      with open(self.json_file, "w") as training_data_file:
        training_data_file.write(json.dumps(self.parsed_json, 
                                            indent=4, separators=(',', ': ')))
    except IOError:
      print "File " + self.json_file + ""

  def noun_chunks_to_json(self):
    for np in self.spacy_doc.noun_chunks:
      if(not self._text_in_common_examples(np.orth_)):
        self._add_text_to_common_examples(np.orth_)

  def verb_chunks_to_json(self):
    vp_elems = self._get_verb_chunks()
    for vp in vp_elems:
      if(not self._intent_in_common_examples(vp)):
        self._add_intent_to_common_examples(vp)

  def sents_to_json(self):
    for s in self.spacy_doc.sents:
      self._add_text_to_common_examples(s)

  def _add_text_to_common_examples(self,phrase):
    self.parsed_json['rasa_nlu_data']['common_examples'].append({
        'text': unicode(phrase),
        'intent':u"", 'entities':[]})

  def _add_intent_to_common_examples(self, phrase):
    self.parsed_json['rasa_nlu_data']['common_examples'].append({
        'intent': unicode(self._phrase_as_intent(phrase)),
        'text':u"", 'entities':[]})

  def _get_verb_chunks(self):
    # naive implementation for verb phrase extraction, can do better
    from spacy.symbols import dobj
    labels = set([dobj])
    elems = []
    for s in self.spacy_doc.sents:
      subandobj = [c for c in s.root.children if(c.dep in labels)]
      if(len(subandobj) > 0):
        elems.append([s.root, subandobj[0]]) # verb, than object
    return elems

  # phrase in this case is a string
  def _text_in_common_examples(self, phrase):
    for ce in self.parsed_json['rasa_nlu_data']['common_examples']:
      if(ce['text'] == phrase):
        return True
    return False

  # phrase is a list of span objects
  def _intent_in_common_examples(self, phrase):
    for ce in self.parsed_json['rasa_nlu_data']['common_examples']:
      if(ce['intent'] == self._phrase_as_intent(phrase)):
        return True
    return False

  def _phrase_as_str(self,phrase):
    return u" ".join([word.orth_.lower() for word in phrase])
  def _phrase_as_intent(self,phrase):
    return u"_".join([word.orth_.lower() for word in phrase])
    
if __name__ == "__main__":
  import sys
  import io
  training_data_file = sys.argv[1]
  input_data_file = sys.argv[2]
  data = str()
  with io.open(input_data_file, "r", encoding='utf-8') as input_data:
    data = input_data.read()
  #TRAINING_DATA_FILE_NAME = "/home/jhughes/dev/my-first-bot/data/my_first_bot.json"
  #MSG = u'Add some text, and we\'ll try and figure out if we can add anything useful to Rasa\'s training data.'
  #s2r = Text2Rasa(MSG, TRAINING_DATA_FILE_NAME)
  s2r = Text2Rasa(data, training_data_file)
  s2r.do_spacy_parse()
  s2r.read_training_file()
  s2r.noun_chunks_to_json()
  s2r.verb_chunks_to_json()
  #s2r.sents_to_json()
  s2r.write_training_file()
  print json.dumps(s2r.parsed_json, indent=4, separators=(',', ': '))
