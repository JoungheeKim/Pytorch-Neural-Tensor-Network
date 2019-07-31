import extractor.subject_verb_object_extract as extract


def findSVOs(tokens):
    svos = []
    is_pas = extract._is_passive(tokens)
    verbs = [tok for tok in tokens if extract._is_non_aux_verb(tok)]
    for v in verbs:
        subs, verbNegated = extract._get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            isConjVerb, conjV = extract._right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = extract._get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = extract._is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(obj, tokens))],
                                             ["not ", v.lemma_] if verbNegated or objNegated else [v.lemma_],
                                             [item.lemma_ for item in (extract.expand(sub, tokens))]))
                            except Exception:
                                pass
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(obj, tokens))],
                                             ["not ", v2.lemma_] if verbNegated or objNegated else [v2.lemma_],
                                             [item.lemma_ for item in (extract.expand(sub, tokens))]))
                            except Exception:
                                pass
                        else:
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(sub, tokens))],
                                             ["not ", v.lemma_] if verbNegated or objNegated else [v.lemma_],
                                             [item.lemma_ for item in (extract.expand(obj, tokens))]))
                            except Exception:
                                pass
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(sub, tokens))],
                                             ["not ", v2.lemma_] if verbNegated or objNegated else [v2.lemma_],
                                             [item.lemma_ for item in (extract.expand(obj, tokens))]))
                            except Exception:
                                pass

            else:
                v, objs = extract._get_all_objs(v, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = extract._is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(obj, tokens))],
                                             ["not ", v.lemma_] if verbNegated or objNegated else [v.lemma_],
                                             [item.lemma_ for item in (extract.expand(sub, tokens))]))
                            except Exception:
                                pass
                        else:
                            try:
                                svos.append(([item.lemma_ for item in (extract.expand(sub, tokens))],
                                             ["not ", v.lemma_] if verbNegated or objNegated else [v.lemma_],
                                             [item.lemma_ for item in (extract.expand(obj, tokens))]))
                            except Exception:
                                pass

    return svos