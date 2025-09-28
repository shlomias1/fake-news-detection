# src/service/predict_cli.py
import argparse, json, sys
from pathlib import Path
from service.predictor import FakeNewsPredictor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--thresholds_json", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--source", default="UNK")
    ap.add_argument("--category", default="UNK")
    ap.add_argument("--date", default="")
    ap.add_argument("--batch_jsonl", help="קלט JSONL: שורות עם title,text,source,category,date")
    ap.add_argument("--out", help="נתיב פלט (JSONL); אם לא יינתן – מדפיס למסך")
    args = ap.parse_args()

    pred = FakeNewsPredictor(args.model_dir, args.thresholds_json)

    out_lines = []
    if args.batch_jsonl:
        for line in Path(args.batch_jsonl).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            res = pred.predict_one(
                title=rec.get("title",""),
                text=rec.get("text",""),
                meta={"source":rec.get("source"), "category":rec.get("category"), "date_published":rec.get("date")}
            )
            out_lines.append(json.dumps({**rec, **res}, ensure_ascii=False))
    else:
        res = pred.predict_one(
            title=args.title,
            text=args.text,
            meta={"source":args.source, "category":args.category, "date_published":args.date}
        )
        out_lines.append(json.dumps(res, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    else:
        for l in out_lines: print(l)

if __name__ == "__main__":
    main()