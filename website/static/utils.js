export function timeFormatter(number, current = "ns") {
    const units = ["ns", "µs", "ms", "s"];
    const factors = [1, 1000, 1000, 1000];

    let currentIndex = units.indexOf(current);
    if (currentIndex === -1) {
        throw new Error("Invalid time");
    }

    while (number < 1 && currentIndex > 0) {
        number *= factors[currentIndex];
        currentIndex--;
    }

    while (number >= factors[currentIndex + 1]) {
        number /= factors[currentIndex + 1];
        currentIndex++;
    }

    return `${number.toFixed(1)}${units[currentIndex]}`;
}

export function showChart(div, series, color) {
    var chart = echarts.init(div, null, {
        // renderer: "svg",
    });

    chart.on("mouseover", () =>
        chart.setOption({ tooltip: { trigger: "item" } })
    );

    chart.on("mouseout", () =>
        chart.setOption({ tooltip: { trigger: "axis" } })
    );

    const lineStyle = {
        color: "#505050",
        width: 1,
    };

    const textStyle = {
        color: getComputedStyle(document.documentElement).getPropertyValue(
            "--main-color"
        ),
        fontFamily: "Fira Sans",
        fontSize: 14,
    };

    /** @type EChartsOption */
    const option = {
        series,
        xAxis: {
            min: "dataMin",
            type: "log",
            name: "Particle Count",
            nameLocation: "middle",
            nameGap: 30,
            axisTick: {
                show: false,
            },
            axisLine: {
                lineStyle,
            },
            axisPointer: {
                snap: false,
                label: {
                    formatter: ({ value }) => value.toFixed(0),
                },
            },
            splitLine: {
                show: true,
                lineStyle,
            },
        },
        yAxis: {
            type: "log",
            name: "Time per iteration",
            nameLocation: "middle",
            nameGap: 70,
            splitLine: {
                show: true,
                lineStyle,
            },
            axisTick: {
                show: false,
            },
            axisLine: {
                lineStyle,
            },
            axisLabel: {
                formatter: (value) => timeFormatter(value),
            },
            axisPointer: {
                label: {
                    formatter: ({ value }) => timeFormatter(value),
                },
            },
        },
        grid: {
            top: "8%",
            bottom: "7%",
            left: "7%",
            right: "14%",
        },
        legend: {
            orient: "vertical",
            top: "center",
            right: "0.75%",
            itemWidth: 10,
            textStyle: {
                ...textStyle,
                fontSize: 12,
            },
        },
        tooltip: {
            trigger: "axis",
            axisPointer: {
                type: "cross",
            },
            backgroundColor: "rgba(10, 10, 10, 0.7)",
            borderWidth: 0,
            valueFormatter: timeFormatter,
            transitionDuration: 0,
            position: (pos, params, dom, rect, { viewSize, contentSize }) => {
                // When data is hovered
                if (!Array.isArray(params)) {
                    return "top";
                }

                const pastCenter = pos[0] < viewSize[0] / 2;
                const offsetX = pastCenter ? 10 : -(10 + contentSize[0]);
                const offset = [offsetX, -30];
                let min = [0, 0];

                params.forEach(({ value }) => {
                    if (value[1] > min[1]) {
                        min = value;
                    }
                });

                const converted = chart.convertToPixel({ seriesIndex: 0 }, min);
                return [converted[0] + offset[0], converted[1] + offset[1]];
            },
            textStyle: {
                ...textStyle,
                fontSize: 12,
            },
            order: "valueDesc",
        },
        toolbox: {
            top: "1%",
            right: "1%",
            iconStyle: {
                borderWidth: 2,
            },
            feature: {
                saveAsImage: {},
                dataView: {
                    backgroundColor: "#1b1b1b",
                    textareaColor: "#121212",
                    textColor: textStyle.color,
                    readOnly: true,
                    title: 'Data',
                    lang: ['Data', 'Close', 'Refresh'],
                },
                dataZoom: {
                    title: {
                        back: "Undo Zoom",
                    },
                },
                restore: {},
            },
        },
        animationDuration: 200,
        textStyle,
        color,
    };

    chart.setOption(option);
}
