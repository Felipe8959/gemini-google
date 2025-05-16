def extract_text_stats(texts):
    ...
    for text in texts:
        ...
        # --- NOVAS FEATURES ---
        num_contact_times = len(re.findall(r"\\b\\d{2}:\\d{2}\\b", text))
        num_phone_calls   = len(re.findall(r"lig(a|ação)|chamada|telefone", text, re.I))
        contact_success   = int(bool(re.search(r"contato (realizado|efetuado|concluído)|cliente atendeu", text, re.I)))
        contact_fail      = int(bool(re.search(r"não (foi possível|houve) contato|sem sucesso|não atendeu", text, re.I)))
        email_count       = len(re.findall(r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+", text))
        attachment_flag   = int(bool(re.search(r"anexo|segue (arquivo|print)|\\.pdf|\\.xlsx|\\.png", text, re.I)))
        conclusion_ct     = len(re.findall(r"resolvid[oa]|concluíd[oa]|encerrad[oa]|solucionad[oa]", text, re.I))
        pending_ct        = len(re.findall(r"aguardando|pendente|em análise|não possível", text, re.I))
        digit_ratio       = sum(ch.isdigit() for ch in text) / len(text) if text else 0
        has_cas           = int(all(kw in text.lower() for kw in [\"causa\", \"ação\", \"solução\"]))\n        \n        features.append([\n            ...  # 23 features anteriores\n            num_contact_times, num_phone_calls, contact_success, contact_fail,\n            email_count, attachment_flag, conclusion_ct, pending_ct,\n            digit_ratio, has_cas\n        ])\n    ..."
