"""Private playlist and readable order report; no audio is copied."""
from pathlib import Path
import html
import hashlib

from .storage import private_path, write_json


def export_order_pack(records, plan, directory):
    directory = private_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {str(r.get("id") or r.get("path")): r for r in records}
    write_json(directory / "harmonize_plan.json", plan)
    write_json(directory / "order.json", plan["order"])
    write_json(directory / "analysis_snapshot.json", records)
    snapshot_hash = hashlib.sha256((directory / "analysis_snapshot.json").read_bytes()).hexdigest()
    write_json(directory / "experiment.json", {"version": plan['version'], 'analysis_snapshot_sha256': snapshot_hash,
        'settings': plan['settings'], 'constraints': plan['constraints'],
        'order_objective': plan.get('order_objective', 'weighted'),
        'objective_details': plan.get('objective_details'),
        'optimal_for_declared_objective': plan['optimal_for_declared_objective'],
        'input_ids': list(by_id), 'chosen_order': plan['order'],
        'audio_copied': False, 'audio_rendered': False,
        'djstudio_result_status': 'awaiting actual export; no comparison with DJ.Studio has yet been performed'})

    def playlist(identities, name):
        if not all(by_id[i].get("path") for i in identities):
            return None
        lines = ["#EXTM3U"]
        for identity in identities:
            path = str(by_id[identity]["path"])
            if "\n" in path or "\r" in path:
                raise ValueError("Playlist paths must not contain newlines")
            # Absolute paths can contain Unicode, # and spaces without encoding.
            lines.extend([f"#EXTINF:-1,{Path(path).stem}", path])
        target = directory / name
        target.write_text("\n".join(lines)+"\n", encoding="utf-8-sig")
        return str(target)

    inputs = playlist(list(by_id), "input_tracks.m3u8")
    ordered = playlist(plan["order"], "harmonized_tracks.m3u8")
    text = ["AI Super DJ · экспериментальный порядок BPM + Key", "",
            f"Настройки: {plan['settings']}",
            f"Поиск: {plan.get('order_objective', 'weighted')} · {plan['method']}",
            f"Дополнительная взвешенная оценка BPM/key: {plan['total_cost']:.6f}",
            f"Пары с низким теоретическим риском: {plan['lower_risk_matches']}/{len(plan['edges'])}",
            f"Дополнительные разрешённые связи: {plan['adventurous_matches']}", ""]
    table = []
    for position, identity in enumerate(plan["order"]):
        row = by_id[identity]
        title = Path(row["path"]).stem if row.get("path") else identity
        edge = plan["edges"][position-1] if position else None
        bpm = row.get("bpm")
        key = row.get("key_camelot") or row.get("key") or "?"
        relation = edge["harmonic"]["relation"] if edge else "—"
        jump = edge["tempo"]["delta_bpm"] if edge else None
        cost = f"{edge['cost']:.4f}" if edge else "—"
        text.append(f"{position+1:02d}. {title} · {bpm} BPM · {key} · {relation}")
        table.append("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in
            (position+1, title, bpm if bpm is not None else "?", key, relation,
             f"{jump:+.3f}" if jump is not None else "—", cost))+"</tr>")
    (directory / "harmonized_tracklist.txt").write_text("\n".join(text)+"\n", encoding="utf-8")
    links = ('<a href="input_tracks.m3u8">Те же исходники для DJ.Studio</a> · '
             '<a href="harmonized_tracks.m3u8">Наш порядок</a>') if inputs else ""
    search_note = ("Сначала минимизируем неподходящие связи тональностей, "
                   + ("затем рискованные связи, " if plan['settings']['mode'] == 'fuzzy' else "")
                   + "затем стоимость темпа и тональности. "
                   if plan.get('order_objective') == 'compatible_first' else "Прежняя взвешенная сумма темпа и тональности. ")
    if plan.get('order_objective') == 'mode_pool':
        search_note = "Обученная модель Fuzzy/Mood выбирает следующий трек с учётом оставшейся музыки. Гармонический отсев независим. Малый остаток проверяется заранее; отдельная ограниченная пересборка исправляет несовместимые стыки всего маршрута, сохраняя исходные решения в журнале. Оставшиеся несовместимые связи отмечены для проверки. Баланс BPM/key и допуск BPM используются для резервного выбора, диагностической стоимости и вторичного критерия пересборки; обученные предпочтения ими не меняются. Направление BPM также влияет на автоматический выбор первого трека. "
    search_note += ("Минимум заявленного критерия доказан." if plan['optimal_for_declared_objective'] else
                    "Поиск ограничен; глобальный минимум не доказан.")
    search_note = html.escape(search_note)
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Super DJ · BPM и тональность</title><style>body{{font:16px/1.6 system-ui;background:#121923;color:#e8eef5;margin:0}}main{{max-width:1200px;margin:auto;padding:28px}}a{{color:#9bcaff}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #38475b}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}</style><main><h1>AI Super DJ · порядок BPM + Key</h1><p>{len(plan['order'])} треков. Настройки: <code>{html.escape(str(plan['settings']))}</code>.</p><p>{links}</p><p>Низкий теоретический гармонический риск: {plan['lower_risk_matches']} из {len(plan['edges'])} пар. Дополнительные разрешённые связи: {plan['adventurous_matches']}. Неизвестные тональности: {plan['harmonic_unknown']}. Максимальный скачок темпа: {plan['maximum_bpm_jump']} BPM.</p><p>Собственная экспериментальная реализация по опубликованным принципам. Таблица связей, веса и поиск принадлежат AI Super DJ; совпадение с закрытым алгоритмом DJ.Studio не установлено. Порядок оценён без энергии, вокала и точек сведения.</p><div class="scroll"><table><tr><th>№</th><th>Трек</th><th>BPM</th><th>Тональность</th><th>Связь с предыдущим</th><th>Δ BPM</th><th>Стоимость пары</th></tr>{''.join(table)}</table></div><h2>Проверка в DJ.Studio</h2><p>Загрузите input_tracks.m3u8. Выберите тот же профиль Fuzzy/Mood, баланс BPM/Key, направление темпа и фиксации. Сохраните полученный порядок в M3U8, CUE или бэкапе DJS. Сохраните также скриншот настроек и версию программы: шкала баланса у нас собственная, точное соответствие положения ползунка ещё предстоит установить.</p><p>Все треки оптимизированы вместе. Для честного теста не фиксируйте прежние четыре папки как отдельные блоки. Фиксация соседств возможна в нашем API через locked_chains.</p><p>У многих исходников тональность определена неуверенно. Сравним оба порядка на одинаковых признаках, а затем, если доступны BPM/Key DJ.Studio, отделим расхождения анализа от расхождений поиска. Разные хорошие порядки допустимы.</p><p><a href="harmonize_plan.json">Полный план и объяснения</a> · <a href="harmonized_tracklist.txt">Треклист</a></p></main></html>'''
    page = page.replace('<p>Собственная экспериментальная', f'<p>{search_note}</p><p>Собственная экспериментальная', 1)
    page = page.replace('У многих исходников тональность определена неуверенно.',
                        'Если исходная тональность определена неуверенно, сначала проверим анализ.')
    if plan.get('order_objective') == 'mode_pool':
        page = page.replace('Все треки оптимизированы вместе.',
                            'Следующий трек выбирается последовательно с учётом оставшихся; глобальный оптимум полного порядка не заявлен.')
        page = page.replace('Поиск ограничен; глобальный минимум не доказан.',
                            'Модель оценивает варианты продолжения; её численная полезность не означает качество звучания.')
    target = directory / "harmonize_report.html"
    target.write_text(page, encoding="utf-8")
    return {"plan": str(directory / "harmonize_plan.json"), "input_playlist": inputs,
            "ordered_playlist": ordered, "report": str(target)}
