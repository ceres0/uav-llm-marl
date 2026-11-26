def test_alg_config_supports_reward(args):
    """
    Check whether algorithm supports specified reward configuration
    """
    if args.common_reward:
        # all algorithms support common reward
        return True
    else:
        if args.learner == "coma_learner" or args.learner == "qtran_learner":
            # COMA and QTRAN only support common reward
            return False
        elif args.learner in ("q_learner", "nq_learner") and (
            args.mixer in ("vdn", "qmix", "oa_qmix")
        ):
            # VDN and QMIX / Finetuned-QMIX only support common reward
            return False
        else:
            return True
