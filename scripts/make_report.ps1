param(
    [string]$InputPath = "reports/report.md",
    [string]$OutputPath = "reports/report.pdf"
)

if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
    throw "pandoc не установлен. установите pandoc или экспортируйте reports/report.md в pdf из редактора."
}

pandoc $InputPath -o $OutputPath --pdf-engine=xelatex
