<#
.SYNOPSIS  Export every slide of a .pptx to PNG using PowerPoint (Windows only) for visual QA.
.EXAMPLE   powershell -File scripts/render_slides.ps1 outputs/Sarcoma/Sarcoma.pptx outputs/Sarcoma/preview
#>
param([Parameter(Mandatory)][string]$Pptx, [Parameter(Mandatory)][string]$OutDir, [int]$Width = 1600)
$Pptx = (Resolve-Path $Pptx).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($Pptx, $true, $false, $false)   # read-only, untitled, no window
    $height = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    foreach ($slide in $pres.Slides) {
        $slide.Export((Join-Path $OutDir ("slide_{0:D2}.png" -f $slide.SlideIndex)), "PNG", $Width, $height)
    }
    $n = $pres.Slides.Count
    $pres.Close()
    "Exported $n slides to $OutDir"
} finally { $app.Quit() }
