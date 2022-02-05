from dataset import get_mask

def test_get_mask():
    input = "/hoge/fuga/Test001/001.tif"
    expect = "/hoge/fuga/Test001_gt/001.bmp"
    actual = get_mask(input)
    assert expect == actual
    
    input = "/hoge/fuga/Test002/002.tif"
    expect = "/hoge/fuga/Test002_gt/002.bmp"
    actual = get_mask(input)
    assert expect == actual
        
    input = "/hoge/fuga/Train002/002.tif"
    expect = "/hoge/fuga/Train002_gt/002.bmp"
    actual = get_mask(input)
    assert expect == actual
    
    input = "/hoge/fuga/piyo/Train002/002.tif"
    expect = "/hoge/fuga/piyo/Train002_gt/002.bmp"
    actual = get_mask(input)
    assert expect == actual