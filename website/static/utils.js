export function timeFormatter(number, current = "ns") {
    const units = ["ns", "µs", "ms", "s", "min"];
    const factors = [1, 1000, 1000, 1000, 60];

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

export function drawChart({ name, benchmarks }, elementId, color) {
    const result = Object.values(benchmarks).reduce((rv, benchmark) => {
        const {
            criterion_benchmark_v1: { function_id, value_str },
            criterion_estimates_v1: {
                mean: { point_estimate },
            },
        } = benchmark;

        rv[function_id] ??= [];
        rv[function_id].push([+value_str, point_estimate]);
        return rv;
    }, {});

    var chart = echarts.init(document.getElementById(elementId), null, {
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
        series: Object.entries(result).map(([label, data]) => {
            return {
                name: label,
                data: [...data].sort(([c1], [c2]) => c1 - c2),
                type: "line",
                symbol: "circle",
                symbolSize: 8,
            };
        }),
        title: {
            text: name,
            textStyle: {
                ...textStyle,
                fontSize: "24",
            },
            top: "1%",
            left: "1%",
        },
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
            position: "top",

            textStyle: {
                ...textStyle,
                fontSize: 12,
            },
            order: "valueAsc",
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
                    textareaColor: "#1b1b1b",
                    textColor: textStyle.color,
                    readOnly: true,
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
        color: color,
    };

    chart.setOption(option);

    console.log(chart.getOption().tooltip);
}
